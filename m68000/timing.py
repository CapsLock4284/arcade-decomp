# license:BSD-3-Clause
"""
Name:        timing.py
Description: how many clock cycles each 68000 instruction takes, for a game whose profile asks for its timing
             (TIMING_MODEL: the board's frames, the program's slowdown)
Version:     0.2.0
Created:     11-11-2026

cycles(instruction) is (base, extras): base the cycles the instruction takes the way the block counts it, extras
{way: more cycles} for the instructions whose time depends on where they go:
  - Bcc: the block counts the shorter way; .b (and .s) taken 10, not taken 8: {"taken": 2}; .w taken 10, not
    taken 12: {"not_taken": 2}
  - DBcc: the block counts the loop (the counter not out, the condition false: 10); {"condition": 2} when the
    condition ends it (12), {"expired": 4} when the counter runs out (14)
The rest take the same time every way they go.
A timed game's lift (m68000/lift.py) adds what the data decides at run time (Effect m68k_cycles): MULU/MULS 2 a
counted bit of the source, DIVU/DIVS their whole time (Jorge Cwik's algorithm of the 68000's microcode), a shift by a
register 2 a bit, BCHG/BCLR/BSET by a register 2 more for the bits 16-31; base_cycles counts the rest of them.
interrupt_check(instruction) is where in the instruction the 68000 decides to take an interrupt at its end: (k, k_way)
cycles from its start. The request has to have come by then; one that comes later is taken at the end of the next
instruction. k_way is None but for Bcc and DBcc: their other way's point is k_way plus its extra cycles. The 68000
samples its interrupt lines at one step of each instruction's microcode, and a request reaches it only when a bus
cycle ends (MAME's core, its m68000.lst: the point is the end of the last bus cycle before that step, mostly the
start of the instruction's last prefetch, 4 cycles before its end). INTERRUPT_POINTS lists the forms that differ:
("late", n) n cycles before the end, ("k", n) n cycles from the start (a time the data decides comes after it).
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
MOVE_DESTINATION["predec"] = (8, 12)    # (MOVE.L to -(An) takes 12, as to (An): MAME's core, measured)
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
DIVIDE = ("divu", "divs")


# the forms whose interrupt point is not 4 cycles before their end (MAME's m68000.lst, BSD-3-Clause, worked out from
# its microcode): "mnemonic.size:source,destination"
INTERRUPT_POINTS = {
    ('k', 0): (
        'divs.w:dreg,dreg', 'reset.:',
    ),
    ('k', 4): (
        'asl.b:dreg,dreg', 'asl.b:quick,dreg', 'asl.l:dreg,dreg', 'asl.l:quick,dreg', 'asl.w:dreg,dreg',
        'asl.w:quick,dreg', 'asr.b:dreg,dreg', 'asr.b:quick,dreg', 'asr.l:dreg,dreg', 'asr.l:quick,dreg',
        'asr.w:dreg,dreg', 'asr.w:quick,dreg', 'bchg.:dreg,dreg', 'bclr.:dreg,dreg', 'bset.:dreg,dreg',
        'divs.w:immediate,dreg', 'divs.w:indirect,dreg', 'divs.w:postinc,dreg', 'lsl.b:dreg,dreg',
        'lsl.b:quick,dreg', 'lsl.l:dreg,dreg', 'lsl.l:quick,dreg', 'lsl.w:dreg,dreg', 'lsl.w:quick,dreg',
        'lsr.b:dreg,dreg', 'lsr.b:quick,dreg', 'lsr.l:dreg,dreg', 'lsr.l:quick,dreg', 'lsr.w:dreg,dreg',
        'lsr.w:quick,dreg', 'muls.w:dreg,dreg', 'mulu.w:dreg,dreg', 'rol.b:dreg,dreg', 'rol.b:quick,dreg',
        'rol.l:dreg,dreg', 'rol.l:quick,dreg', 'rol.w:dreg,dreg', 'rol.w:quick,dreg', 'ror.b:dreg,dreg',
        'ror.b:quick,dreg', 'ror.l:dreg,dreg', 'ror.l:quick,dreg', 'ror.w:dreg,dreg', 'ror.w:quick,dreg',
        'roxl.b:dreg,dreg', 'roxl.b:quick,dreg', 'roxl.l:dreg,dreg', 'roxl.l:quick,dreg', 'roxl.w:dreg,dreg',
        'roxl.w:quick,dreg', 'roxr.b:dreg,dreg', 'roxr.b:quick,dreg', 'roxr.l:dreg,dreg', 'roxr.l:quick,dreg',
        'roxr.w:dreg,dreg', 'roxr.w:quick,dreg',
    ),
    ('k', 6): (
        'divs.w:predec,dreg',
    ),
    ('k', 8): (
        'bchg.:immediate,dreg', 'bclr.:immediate,dreg', 'bset.:immediate,dreg', 'divs.w:abs_short,dreg',
        'divs.w:disp,dreg', 'divs.w:pc_disp,dreg', 'muls.w:immediate,dreg', 'muls.w:indirect,dreg',
        'muls.w:postinc,dreg', 'mulu.w:immediate,dreg', 'mulu.w:indirect,dreg', 'mulu.w:postinc,dreg',
    ),
    ('k', 10): (
        'divs.w:index,dreg', 'divs.w:pc_index,dreg', 'muls.w:predec,dreg', 'mulu.w:predec,dreg',
    ),
    ('k', 12): (
        'divs.w:abs_long,dreg', 'muls.w:abs_short,dreg', 'muls.w:disp,dreg', 'muls.w:pc_disp,dreg',
        'mulu.w:abs_short,dreg', 'mulu.w:disp,dreg', 'mulu.w:pc_disp,dreg',
    ),
    ('k', 14): (
        'muls.w:index,dreg', 'muls.w:pc_index,dreg', 'mulu.w:index,dreg', 'mulu.w:pc_index,dreg',
    ),
    ('k', 16): (
        'muls.w:abs_long,dreg', 'mulu.w:abs_long,dreg',
    ),
    ('late', 2): (
        'abcd.:dreg,dreg', 'add.l:abs_long,dreg', 'add.l:abs_short,dreg', 'add.l:disp,dreg', 'add.l:index,dreg',
        'add.l:indirect,dreg', 'add.l:pc_disp,dreg', 'add.l:pc_index,dreg', 'add.l:postinc,dreg',
        'add.l:predec,dreg', 'adda.l:abs_long,areg', 'adda.l:abs_short,areg', 'adda.l:disp,areg',
        'adda.l:index,areg', 'adda.l:indirect,areg', 'adda.l:pc_disp,areg', 'adda.l:pc_index,areg',
        'adda.l:postinc,areg', 'adda.l:predec,areg', 'and.l:abs_long,dreg', 'and.l:abs_short,dreg',
        'and.l:disp,dreg', 'and.l:index,dreg', 'and.l:indirect,dreg', 'and.l:pc_disp,dreg', 'and.l:pc_index,dreg',
        'and.l:postinc,dreg', 'and.l:predec,dreg', 'btst.:dreg,dreg', 'btst.:dreg,immediate', 'btst.:immediate,dreg',
        'clr.l:dreg', 'cmp.l:abs_long,dreg', 'cmp.l:abs_short,dreg', 'cmp.l:areg,dreg', 'cmp.l:disp,dreg',
        'cmp.l:dreg,dreg', 'cmp.l:immediate,dreg', 'cmp.l:index,dreg', 'cmp.l:indirect,dreg', 'cmp.l:pc_disp,dreg',
        'cmp.l:pc_index,dreg', 'cmp.l:postinc,dreg', 'cmp.l:predec,dreg', 'cmpa.l:abs_long,areg',
        'cmpa.l:abs_short,areg', 'cmpa.l:areg,areg', 'cmpa.l:disp,areg', 'cmpa.l:dreg,areg', 'cmpa.l:immediate,areg',
        'cmpa.l:index,areg', 'cmpa.l:indirect,areg', 'cmpa.l:pc_disp,areg', 'cmpa.l:pc_index,areg',
        'cmpa.l:postinc,areg', 'cmpa.l:predec,areg', 'cmpa.w:abs_long,areg', 'cmpa.w:abs_short,areg',
        'cmpa.w:areg,areg', 'cmpa.w:disp,areg', 'cmpa.w:dreg,areg', 'cmpa.w:immediate,areg', 'cmpa.w:index,areg',
        'cmpa.w:indirect,areg', 'cmpa.w:pc_disp,areg', 'cmpa.w:pc_index,areg', 'cmpa.w:postinc,areg',
        'cmpa.w:predec,areg', 'cmpi.l:immediate,dreg', 'exg.:areg,areg', 'exg.:dreg,areg', 'exg.:dreg,dreg',
        'move.:sr,dreg', 'nbcd.:dreg', 'neg.l:dreg', 'negx.l:dreg', 'not.l:dreg', 'or.l:abs_long,dreg',
        'or.l:abs_short,dreg', 'or.l:disp,dreg', 'or.l:index,dreg', 'or.l:indirect,dreg', 'or.l:pc_disp,dreg',
        'or.l:pc_index,dreg', 'or.l:postinc,dreg', 'or.l:predec,dreg', 'sbcd.:dreg,dreg', 'sub.l:abs_long,dreg',
        'sub.l:abs_short,dreg', 'sub.l:disp,dreg', 'sub.l:index,dreg', 'sub.l:indirect,dreg', 'sub.l:pc_disp,dreg',
        'sub.l:pc_index,dreg', 'sub.l:postinc,dreg', 'sub.l:predec,dreg', 'suba.l:abs_long,areg',
        'suba.l:abs_short,areg', 'suba.l:disp,areg', 'suba.l:index,areg', 'suba.l:indirect,areg',
        'suba.l:pc_disp,areg', 'suba.l:pc_index,areg', 'suba.l:postinc,areg', 'suba.l:predec,areg',
    ),
    ('late', 6): (
        'lea.:index,areg', 'lea.:pc_index,areg', 'trap.:quick',
    ),
}
POINT_OF_FORM = {form: rule for rule, forms in INTERRUPT_POINTS.items() for form in forms}


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
        n = count.value if count.mode in ("quick", "immediate") else 0    # (by a register: the lift adds 2 a bit)
        return (8 if long else 6) + 2 * n
    if name in ("swap", "ext", "nop", "trapv", "illegal", "stop"):
        return 4
    if name == "exg":
        return 6
    if name in ("btst", "bchg", "bclr", "bset"):
        static = ops[0].mode == "immediate"
        target = ops[1]
        if target.mode == "dreg":
            high = 2 if static and (ops[0].value & 31) >= 16 else 0    # (bits 16-31; by a register the lift adds it)
            if name == "btst":
                return 10 if static else 6
            if name == "bclr":
                return (12 if static else 8) + high
            return (10 if static else 6) + high
        if name == "btst":
            return (8 if static else 4) + ea(target, False)
        return (12 if static else 8) + ea(target, False)
    if name in ("mulu", "muls"):
        return 38 + ea(ops[0], False)    # (+ 2 a counted bit: the lift adds it)
    if name in DIVIDE:
        return ea(ops[0], False)         # (the division's own time: the lift adds it)
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


def interrupt_check(instruction):
    """(k, k_way): the cycles from the instruction's start to the point an interrupt request has to have come by to
    be taken at its end (the block's way); k_way for a Bcc's or DBcc's other way, plus that way's extra cycles"""
    name = instruction.mnemonic
    base = base_cycles(instruction)
    if name == "bcc":
        if instruction.length == 2:
            return 0, 4    # (.b: not taken 8, the request by its start; taken 10, by 6)
        return 6, 6        # (.w: taken 10, by 6; not taken 12, by 8)
    if name == "bra":
        return 6, None
    if name == "dbcc":
        return 6, 6        # (10, 12, 14: 4 before the end each)
    if name in ("divu", "chk"):
        return ea(instruction.operands[0], False), None    # (the division's or the check's own time after it)
    form = f"{name}.{instruction.size or ''}:{','.join(operand.mode for operand in instruction.operands)}"
    kind, n = POINT_OF_FORM.get(form, ("late", 4))
    if kind == "k":
        return n, None
    return max(base - n, 0), None
