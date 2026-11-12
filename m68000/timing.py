# license:BSD-3-Clause
"""
Name:        timing.py
Description: how many clock cycles each 68000 instruction takes, for a game whose profile asks for its timing
             (TIMING_MODEL: the board's frames, the program's slowdown)
Version:     0.1.0
Created:     11-11-2026

cycles(instruction) is (base, extras): base the cycles the instruction takes the way the block counts it, extras
{way: more cycles} for the instructions whose time depends on where they go:
  - Bcc: the block counts the shorter way; .b (and .s) taken 10, not taken 8: {"taken": 2}; .w taken 10, not
    taken 12: {"not_taken": 2}
  - DBcc: the block counts the loop (the counter not out, the condition false: 10); {"condition": 2} when the
    condition ends it (12), {"expired": 4} when the counter runs out (14)
The rest take the same time every way they go. Where the time depends on the data, a fixed one is taken:
  - shifts and rotates by a register's count: 6 + 2n (.l 8 + 2n) with n = 4, a typical count (DonPachi's
    shifts by a register are mostly of a few bits)
  - MULU / MULS: 38 + 2n with n = 8, half the bits (the manual's range is 38-70)
  - DIVU 120, DIVS 140 (the manual's most are 140 and 158; most divisions end sooner)
  - BCHG, BCLR, BSET on a data register: as for bit numbers 16-31 (+2), the slower way
These are the Motorola M68000 User's Manual's timings (section 8, "Instruction Execution Times"), which MAME's
core keeps; tools/cref/cref_timed (DonPachi's) measures the frames on MAME's core, to compare.
"""
from . import disassembler as dis

# the effective address's calculation and fetch: (byte or word, long), the manual's table 8-1
EA = {
    "dreg": (0, 0), "areg": (0, 0), "indirect": (4, 8), "postinc": (4, 8), "predec": (6, 10),
    "disp": (8, 12), "index": (10, 14), "abs_short": (8, 12), "abs_long": (12, 16),
    "pc_disp": (8, 12), "pc_index": (10, 14), "immediate": (4, 8),
}
# MOVE: the destination's part, byte or word and long (tables 8-2 and 8-3, the Dn source row)
MOVE_DESTINATION = {
    "dreg": (4, 4), "areg": (4, 4), "indirect": (8, 12), "postinc": (8, 12), "predec": (8, 14),
    "disp": (12, 16), "index": (14, 18), "abs_short": (12, 16), "abs_long": (16, 20),
}
# JMP, JSR, LEA, PEA by their operand's mode (table 8-12)
JMP = {"indirect": 8, "disp": 10, "index": 14, "abs_short": 10, "abs_long": 12, "pc_disp": 10, "pc_index": 14}
JSR = {"indirect": 16, "disp": 18, "index": 22, "abs_short": 18, "abs_long": 20, "pc_disp": 18, "pc_index": 22}
LEA = {"indirect": 4, "disp": 8, "index": 12, "abs_short": 8, "abs_long": 12, "pc_disp": 8, "pc_index": 12}
PEA = {"indirect": 12, "disp": 16, "index": 20, "abs_short": 16, "abs_long": 20, "pc_disp": 16, "pc_index": 20}
# MOVEM (table 8-12): memory to registers, registers to memory; + 4 a word register, + 8 a long one
MOVEM_TO_REGISTERS = {"indirect": 12, "postinc": 12, "disp": 16, "index": 18, "abs_short": 16, "abs_long": 20,
                      "pc_disp": 16, "pc_index": 18}
MOVEM_TO_MEMORY = {"indirect": 8, "predec": 8, "disp": 12, "index": 14, "abs_short": 12, "abs_long": 16}

STANDARD = ("add", "sub", "and", "or", "cmp", "eor")
IMMEDIATE = ("addi", "subi", "andi", "ori", "eori", "cmpi")
SINGLE = ("clr", "neg", "negx", "not")
SHIFTS = ("asl", "asr", "lsl", "lsr", "rol", "ror", "roxl", "roxr")
SHIFT_COUNT = 4         # a shift by a register's count: the count taken
MULTIPLY_BITS = 8       # MULU/MULS: the bits that cost 2 each
DIVIDE = {"divu": 120, "divs": 140}


def ea(operand, long):
    """The cycles an operand's effective address adds (0 for a register)."""
    return EA.get(operand.mode, (0, 0))[1 if long else 0]


def is_register(operand):
    return operand.mode in ("dreg", "areg")


def base_cycles(instruction):
    """The instruction's time, the shorter way for those that branch (Bcc, DBcc's loop)."""
    name = instruction.mnemonic
    size = instruction.size
    long = size == "l"
    ops = instruction.operands
    if name in ("move", "movea"):
        source, destination = ops
        if source.mode in ("sr", "ccr", "usp") or destination.mode in ("sr", "ccr", "usp"):
            if destination.mode in ("sr", "ccr"):
                return 12 + ea(source, False)                 # MOVE to SR / CCR
            if source.mode == "usp" or destination.mode == "usp":
                return 4
            return 6 if is_register(destination) else 8 + ea(destination, False)    # MOVE from SR
        return MOVE_DESTINATION.get(destination.mode, (4, 4))[1 if long else 0] + ea(source, long)
    if name == "moveq":
        return 4
    if name in STANDARD:
        source, destination = ops
        if destination.mode == "dreg":
            if name == "cmp":
                return (6 if long else 4) + ea(source, long)
            if long:
                return (8 if source.mode in ("dreg", "areg", "immediate") else 6) + ea(source, long)
            return 4 + ea(source, long)
        return (12 if long else 8) + ea(destination, long)    # Dn to memory
    if name in ("adda", "suba", "cmpa"):
        source = ops[0]
        if name == "cmpa":
            return 6 + ea(source, long)
        if long:
            return (8 if source.mode in ("dreg", "areg", "immediate") else 6) + ea(source, long)
        return 8 + ea(source, long)
    if name in IMMEDIATE:
        destination = ops[1]
        if destination.mode in ("sr", "ccr"):
            return 20
        if destination.mode == "dreg":
            if name == "cmpi":
                return 14 if long else 8
            return 16 if long else 8
        if name == "cmpi":
            return (12 if long else 8) + ea(destination, long)
        return (20 if long else 12) + ea(destination, long)
    if name in ("addq", "subq"):
        destination = ops[1]
        if destination.mode == "dreg":
            return 8 if long else 4
        if destination.mode == "areg":
            return 8
        return (12 if long else 8) + ea(destination, long)
    if name in ("addx", "subx"):
        if ops[0].mode == "dreg":
            return 8 if long else 4
        return 30 if long else 18
    if name in ("abcd", "sbcd"):
        return 6 if ops[0].mode == "dreg" else 18
    if name == "cmpm":
        return 20 if long else 12
    if name in SINGLE:
        destination = ops[0]
        if destination.mode == "dreg":
            return 6 if long else 4
        return (12 if long else 8) + ea(destination, long)
    if name == "tst":
        return 4 + ea(ops[0], long)
    if name == "nbcd":
        return 6 if ops[0].mode == "dreg" else 8 + ea(ops[0], False)
    if name == "tas":
        return 4 if ops[0].mode == "dreg" else 10 + ea(ops[0], False)
    if name == "scc":
        return 4 if ops[0].mode == "dreg" else 8 + ea(ops[0], False)    # (6 on a register when true)
    if name in SHIFTS:
        if len(ops) == 1:
            return 8 + ea(ops[0], False)                      # memory, by one bit
        count = ops[0]
        n = count.value if count.mode in ("quick", "immediate") else SHIFT_COUNT
        return (8 if long else 6) + 2 * n
    if name in ("swap", "ext", "nop", "trapv", "illegal", "stop"):
        return 4
    if name == "exg":
        return 6
    if name in ("btst", "bchg", "bclr", "bset"):
        static = ops[0].mode == "immediate"
        target = ops[1]
        if target.mode == "dreg":
            if name == "btst":
                return 10 if static else 6
            if name == "bclr":
                return 14 if static else 10
            return 12 if static else 8
        if name == "btst":
            return (8 if static else 4) + ea(target, False)
        return (12 if static else 8) + ea(target, False)
    if name in ("mulu", "muls"):
        return 38 + 2 * MULTIPLY_BITS + ea(ops[0], False)
    if name in DIVIDE:
        return DIVIDE[name] + ea(ops[0], False)
    if name == "chk":
        return 10 + ea(ops[0], False)
    if name == "lea":
        return LEA.get(ops[0].mode, 4)
    if name == "pea":
        return PEA.get(ops[0].mode, 12)
    if name == "jmp":
        return JMP.get(ops[0].mode, 8)
    if name == "jsr":
        return JSR.get(ops[0].mode, 16)
    if name == "bra":
        return 10
    if name == "bsr":
        return 18
    if name == "bcc":
        return 8 if size in ("b", "s", None) and instruction.length == 2 else 10
    if name == "dbcc":
        return 10
    if name == "rts":
        return 16
    if name in ("rte", "rtr"):
        return 20
    if name == "link":
        return 16
    if name == "unlk":
        return 12
    if name == "movem":
        source, destination = ops
        if source.mode == "reglist":
            count = len(source.registers)
            return MOVEM_TO_MEMORY.get(destination.mode, 8) + (8 if long else 4) * count
        count = len(destination.registers)
        return MOVEM_TO_REGISTERS.get(source.mode, 12) + (8 if long else 4) * count
    if name == "movep":
        return 24 if long else 16
    if name == "trap":
        return 34
    if name == "reset":
        return 132
    return 4


def cycles(instruction):
    """(base, extras): the instruction's time the way its block counts it, and the more each other way takes."""
    base = base_cycles(instruction)
    if instruction.mnemonic == "bcc":
        short_form = instruction.length == 2
        return base, ({"taken": 2} if short_form else {"not_taken": 2})
    if instruction.mnemonic == "dbcc":
        return base, {"condition": 2, "expired": 4}
    return base, {}
