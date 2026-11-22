// Tiny stack VM: pre-decoding interpreter with a handler table.
// Pass one decodes bytes into {op, arg} objects keyed by byte address;
// an address the linear sweep missed (jump into an instruction's
// middle) is decoded on demand and cached, staying byte-exact.
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

class Trap extends Error {
  code: number;
  constructor(code: number) { super("trap " + code); this.code = code; }
}
function trap(code: number): never { throw new Trap(code); }

// Immediate byte counts per opcode (absent = 0). An opcode with no
// handler in the table below is unknown and traps 5.
const IMM: Record<number, number> = { 1: 1, 2: 4, 48: 2, 49: 2, 50: 2, 51: 2 };
// [popped, pushed] per opcode, for the under/overflow pre-checks.
const NEED: Record<number, [number, number]> = {
  1: [0, 1], 2: [0, 1], 3: [1, 0], 4: [1, 2], 5: [2, 2], 6: [2, 3],
  7: [3, 3], 23: [1, 1], 27: [1, 1], 49: [1, 0], 50: [1, 0], 64: [1, 1],
  65: [2, 0], 80: [1, 0], 81: [1, 0], 82: [0, 1], 83: [1, 0],
};
for (const op of [16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34]) NEED[op] = [2, 1];

type Ins = { op: number; arg: number; len: number; next: number };
type State = {
  stack: Int32Array; sp: number; mem: Int32Array; calls: number[];
  input: Buffer; inPos: number; out: Buffer[]; outLen: number;
  pc: number; done: boolean; code: number;
};


function decodeAt(prog: Buffer, addr: number): Ins {
  const op = prog[addr];
  const imm = IMM[op] ?? 0;
  let arg = 0;
  if (addr + 1 + imm <= prog.length) {
    if (op === 0x01) arg = (prog[addr + 1] << 24) >> 24;
    else if (op === 0x02) {
      arg = prog[addr + 1] | (prog[addr + 2] << 8) |
        (prog[addr + 3] << 16) | (prog[addr + 4] << 24);
    } else if (imm === 2) {
      const raw = prog[addr + 1] | (prog[addr + 2] << 8);
      arg = op === 0x33 ? raw : (raw << 16) >> 16;
    }
  }
  const len = 1 + imm;
  return { op, arg, len, next: addr + len };
}

function pop(s: State): number { return s.stack[--s.sp]; }
function push(s: State, v: number): void { s.stack[s.sp++] = v | 0; }
function bin(s: State, f: (a: number, b: number) => number): void {
  const b = pop(s); const a = pop(s); push(s, f(a, b));
}
function emit(s: State, chunk: Buffer): void {
  if (s.outLen + chunk.length > 65536) trap(11);
  s.out.push(chunk); s.outLen += chunk.length;
}

const HANDLERS: Record<number, (s: State, ins: Ins) => void> = {
  0x00: (s) => { s.done = true; s.code = 0; },
  0x01: (s, ins) => push(s, ins.arg),
  0x02: (s, ins) => push(s, ins.arg),
  0x03: (s) => { pop(s); }, 0x04: (s) => push(s, s.stack[s.sp - 1]),
  0x05: (s) => { const b = pop(s); const a = pop(s); push(s, b); push(s, a); },
  0x06: (s) => push(s, s.stack[s.sp - 2]),
  0x07: (s) => { // ( a b c -- b c a )
    const a = s.stack[s.sp - 3];
    s.stack[s.sp - 3] = s.stack[s.sp - 2];
    s.stack[s.sp - 2] = s.stack[s.sp - 1];
    s.stack[s.sp - 1] = a;
  },
  0x10: (s) => bin(s, (a, b) => a + b), 0x11: (s) => bin(s, (a, b) => a - b),
  0x12: (s) => bin(s, (a, b) => Math.imul(a, b)),
  0x13: (s) => bin(s, (a, b) => b === 0 ? trap(4) :
    a === -2147483648 && b === -1 ? a : Math.trunc(a / b)),
  0x14: (s) => bin(s, (a, b) => (b === 0 ? trap(4) : a % b)),
  0x15: (s) => bin(s, (a, b) => b === 0 ? trap(4) : Math.trunc((a >>> 0) / (b >>> 0))),
  0x16: (s) => bin(s, (a, b) => b === 0 ? trap(4) : (a >>> 0) % (b >>> 0)),
  0x17: (s) => push(s, -pop(s)), 0x1b: (s) => push(s, ~pop(s)),
  0x18: (s) => bin(s, (a, b) => a & b), 0x19: (s) => bin(s, (a, b) => a | b),
  0x1a: (s) => bin(s, (a, b) => a ^ b),
  0x1c: (s) => bin(s, (a, b) => a << (b & 31)),
  0x1d: (s) => bin(s, (a, b) => a >> (b & 31)),
  0x1e: (s) => bin(s, (a, b) => (a >>> (b & 31)) | 0),
  0x20: (s) => bin(s, (a, b) => (a === b ? 1 : 0)),
  0x21: (s) => bin(s, (a, b) => (a < b ? 1 : 0)),
  0x22: (s) => bin(s, (a, b) => ((a >>> 0) < (b >>> 0) ? 1 : 0)),
  0x30: (s, ins) => { s.pc = ins.next + ins.arg; },
  0x31: (s, ins) => { if (pop(s) === 0) s.pc = ins.next + ins.arg; },
  0x32: (s, ins) => { if (pop(s) !== 0) s.pc = ins.next + ins.arg; },
  0x33: (s, ins) => {
    if (s.calls.length >= 64) trap(7);
    s.calls.push(ins.next); s.pc = ins.arg;
  },
  0x34: (s) => {
    if (s.calls.length === 0) trap(8);
    s.pc = s.calls.pop() as number;
  },
  0x40: (s) => {
    const addr = pop(s);
    if (addr < 0 || addr > 255) trap(6);
    push(s, s.mem[addr]);
  },
  0x41: (s) => {
    const addr = pop(s); const v = pop(s);
    if (addr < 0 || addr > 255) trap(6);
    s.mem[addr] = v;
  },
  0x50: (s) => emit(s, Buffer.from(String(pop(s)), "ascii")),
  0x51: (s) => emit(s, Buffer.from([pop(s) & 255])),
  0x52: (s) => push(s, s.inPos < s.input.length ? s.input[s.inPos++] : -1),
  0x53: (s) => { s.done = true; s.code = pop(s) & 255; },
};

function run(prog: Buffer, input: Buffer): { out: Buffer; code: number } {
  // Pre-decode pass: linear sweep collecting instruction objects.
  const byAddr = new Map<number, Ins>();
  const decoded: Ins[] = [];
  for (let at = 0; at < prog.length;) {
    const ins = decodeAt(prog, at);
    decoded.push(ins); byAddr.set(at, ins);
    if (at + ins.len > prog.length) break;
    at += ins.len;
  }
  const s: State = {
    stack: new Int32Array(256), sp: 0, mem: new Int32Array(256),
    calls: [], input, inPos: 0, out: [], outLen: 0,
    pc: 0, done: false, code: 0,
  };
  let steps = 0;
  try {
    while (!s.done) {
      if (steps >= 1000000) trap(10);
      if (s.pc < 0 || s.pc >= prog.length) trap(9);
      let ins = byAddr.get(s.pc);
      if (ins === undefined) {
        ins = decodeAt(prog, s.pc);
        byAddr.set(s.pc, ins);
        decoded.push(ins);
      }
      const handler = HANDLERS[ins.op];
      if (handler === undefined) trap(5);
      if (s.pc + ins.len > prog.length) trap(9);
      const nd = NEED[ins.op] ?? [0, 0];
      if (s.sp < nd[0]) trap(2);
      if (s.sp - nd[0] + nd[1] > 256) trap(3);
      steps++;
      s.pc = ins.next;
      handler(s, ins);
    }
  } catch (e) {
    if (e instanceof Trap) s.code = e.code;
    else throw e;
  }
  return { out: Buffer.concat(s.out), code: s.code };
}

const inDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
for (const entry of readdirSync(inDir, { withFileTypes: true })) {
  if (!entry.isDirectory()) continue;
  const prog = readFileSync(join(inDir, entry.name, "program.bin"));
  let input = Buffer.alloc(0);
  try { input = readFileSync(join(inDir, entry.name, "input.bin")); }
  catch { /* no input.bin: reads as empty */ }
  const r = run(prog, input);
  const dest = join(outDir, entry.name);
  mkdirSync(dest, { recursive: true });
  writeFileSync(join(dest, "stdout.bin"), r.out);
  writeFileSync(join(dest, "exit.txt"), String(r.code) + "\n");
}
