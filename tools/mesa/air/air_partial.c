/*
 * Copyright © 2026 Darwin on redwood contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * THE STATIC INITIALIZERS, RUN WHILE TRANSLATING.
 *
 * A module's air.static_init functions turn its function constants into the
 * program-scope variables the body reads. CoreAnimation's compositing shader
 * (QuartzCore's fixed_frag_lph_cph and _cpf, the shader every layer of the
 * window server is drawn with) decides 58 of them that way -- among them which
 * of its textures and samplers exist at all. Translated as code, that
 * initializer is a function run for every pixel whose results live in scratch
 * memory, where NIR's constant folding cannot reach them; and the body's
 * choices between textures and between samplers, which this frontend has to
 * resolve while it translates (a binding is not a value), stay choices.
 *
 * With the pipeline's function constants in hand nothing in an initializer is
 * unknown, so it is evaluated here, over the bytes of the variables. A
 * variable nothing but the initializers writes then becomes a constant
 * (air_known): its bytes go to the shader's constant data, where a load at a
 * constant offset is a constant while this frontend translates -- which is
 * what lets a choice of sampler be made here -- and NIR folds the rest. A
 * variable the body also writes keeps its scratch memory and gets the value
 * the initializers left at the top of the entry point, where running them
 * used to be.
 *
 * Anything this evaluator does not know -- an instruction, an intrinsic, a
 * load it has nothing to read from -- gives up, quietly, and the initializers
 * are translated as code as before. It knows what the corpus's initializers
 * are made of: loads of function constants and of variables, integer
 * arithmetic, comparisons, selects and casts, phis, branches and switches, and
 * the two function-constant intrinsics.
 */

#include "air_private.h"

#include <stdio.h>
#include <stdlib.h>

#include "util/hash_table.h"
#include "util/ralloc.h"

struct ev_val {
   bool is_ptr;
   unsigned global;     /* is_ptr: index in m->globals */
   uint64_t offset;     /* is_ptr: bytes into it */
   unsigned n, bits;    /* components, and bits per component */
   uint64_t lane[16];
};

struct ev {
   struct air_ctx *c;
   const struct air_module *m;
   struct air_known *k;
   struct hash_table *index;   /* const struct air_global * -> index + 1 */
   struct hash_table *vals;    /* const struct air_value * -> struct ev_val * */
};

static uint64_t
mask_of(unsigned bits)
{
   return bits >= 64 ? ~0ull : (1ull << bits) - 1;
}

static int
global_index(struct ev *e, const struct air_value *v)
{
   struct hash_entry *he = _mesa_hash_table_search(e->index, v);
   return he ? (int)(uintptr_t)he->data - 1 : -1;
}

/* What plan_globals makes a program-scope variable: everything that is not a
 * linker list, a function constant, an inline sampler or threadgroup memory.
 * Kept in step with it by hand; a variable this says is one and plan_globals
 * does not would simply not be modelled.
 */
static bool
is_program_variable(const struct air_global *g)
{
   const char *name = g->base.name ? g->base.name : "";
   return strncmp(name, "llvm.", 5) && !g->externally_initialized &&
          strncmp(name, "__air_sampler_state", 19) && g->addrspace != 3;
}

static int
function_constant_of(struct air_ctx *c, const struct air_global *g)
{
   return g->externally_initialized ? air_function_constant_index(c, g) : -1;
}

/* the value the pipeline gave a function constant, or none */
static const struct air_constant_value *
fc_value(struct air_ctx *c, unsigned index)
{
   for (unsigned i = 0; c->options && i < c->options->num_constants; i++)
      if (c->options->constants[i].index == index)
         return &c->options->constants[i];
   return NULL;
}

/* -- the variables' bytes ---------------------------------------------- */

static void
add_pointer_cell(struct ev *e, unsigned global, uint64_t offset, unsigned target)
{
   struct air_known *k = e->k;
   for (unsigned i = 0; i < k->num_ptrs; i++) {
      if (k->ptrs[i].global == global && k->ptrs[i].offset == offset) {
         k->ptrs[i].target = target;
         return;
      }
   }
   k->ptrs = reralloc(e->c->mem_ctx, k->ptrs, struct air_known_ptr, k->num_ptrs + 1);
   k->ptrs[k->num_ptrs++] = (struct air_known_ptr){ global, offset, target };
}

/* The base global and byte offset of a pointer constant: a global itself, or a
 * constant getelementptr or cast of one.
 */
static bool
const_pointer(struct ev *e, const struct air_value *v, unsigned *global, uint64_t *offset);

static bool
gep_offset(struct ev *e, const struct air_type *src, struct air_value *const *idx,
           unsigned num, const struct air_value *(*get)(struct ev *, const struct air_value *),
           uint64_t *out)
{
   (void)get;
   uint64_t off = 0;
   const struct air_type *t = src;
   for (unsigned i = 0; i < num; i++) {
      uint64_t x;
      if (!air_const_uint(idx[i], &x))
         return false;
      if (i == 0) {
         off += x * air_type_alloc_size(e->m, src);
         continue;
      }
      if (t->kind == AIR_TYPE_STRUCT) {
         if (x >= t->num_members)
            return false;
         off += air_struct_member_offset(e->m, t, (unsigned)x);
         t = t->members[x];
      } else if (t->kind == AIR_TYPE_ARRAY || t->kind == AIR_TYPE_VECTOR) {
         off += x * air_type_alloc_size(e->m, t->elem);
         t = t->elem;
      } else {
         return false;
      }
   }
   *out = off;
   return true;
}

static bool
const_pointer(struct ev *e, const struct air_value *v, unsigned *global, uint64_t *offset)
{
   if (v->kind == AIR_VALUE_GLOBAL) {
      int gi = global_index(e, v);
      if (gi < 0)
         return false;
      *global = (unsigned)gi;
      *offset = 0;
      return true;
   }
   const struct air_constant *k = air_value_as_constant(v);
   if (!k || k->ckind != AIR_CONST_EXPR || !k->num_operands)
      return false;
   if (k->expr_op == AIR_OP_CAST && (k->expr_sub == AIR_CAST_BITCAST ||
                                     k->expr_sub == AIR_CAST_ADDRSPACECAST))
      return const_pointer(e, k->operands[0], global, offset);
   if (k->expr_op == AIR_OP_GEP) {
      uint64_t base_off, off;
      if (!const_pointer(e, k->operands[0], global, &base_off) ||
          !gep_offset(e, k->src_type, k->operands + 1, k->num_operands - 1, NULL, &off))
         return false;
      *offset = base_off + off;
      return true;
   }
   return false;
}

static void
write_bytes(uint8_t *mem, uint64_t size, uint64_t offset, uint64_t value, unsigned bytes)
{
   for (unsigned b = 0; b < bytes && offset + b < size; b++)
      mem[offset + b] = (uint8_t)(value >> (8 * b));
}

/* A variable's initializer into its bytes. False when it holds something the
 * bytes cannot (a float the reader did not keep, an expression); a pointer to
 * another global is a cell beside the bytes.
 */
static bool
init_bytes(struct ev *e, unsigned gi, uint64_t offset, const struct air_value *v)
{
   struct air_known *k = e->k;
   const struct air_type *t = v->type;
   const struct air_constant *kc = air_value_as_constant(v);

   if (t->kind == AIR_TYPE_POINTER) {
      unsigned target;
      uint64_t toff;
      if (kc && (kc->ckind == AIR_CONST_NULL || kc->ckind == AIR_CONST_UNDEF ||
                 kc->ckind == AIR_CONST_POISON))
         return true;
      if (!const_pointer(e, v, &target, &toff) || toff != 0)
         return false;
      add_pointer_cell(e, gi, offset, target);
      return true;
   }
   if (v->kind == AIR_VALUE_GLOBAL)
      return false;
   if (!kc)
      return false;
   switch (kc->ckind) {
   case AIR_CONST_NULL:
   case AIR_CONST_UNDEF:
   case AIR_CONST_POISON:
      return true;   /* the bytes start as zeros */
   case AIR_CONST_INT:
   case AIR_CONST_FLOAT: {
      unsigned bits = air_type_bit_size(t);
      write_bytes(k->bytes[gi], k->size[gi], offset, kc->u, MAX2(bits, 8) / 8);
      return true;
   }
   case AIR_CONST_DATA: {
      const struct air_type *el = t->elem;
      if (!el || (t->kind != AIR_TYPE_ARRAY && t->kind != AIR_TYPE_VECTOR))
         return false;
      uint64_t stride = air_type_alloc_size(e->m, el);
      unsigned bytes = MAX2(air_type_bit_size(el), 8) / 8;
      for (uint64_t i = 0; i < t->count; i++)
         write_bytes(k->bytes[gi], k->size[gi], offset + i * stride, kc->data[i], bytes);
      return true;
   }
   case AIR_CONST_AGGREGATE:
      for (unsigned i = 0; i < kc->num_elems; i++) {
         uint64_t off;
         if (t->kind == AIR_TYPE_STRUCT)
            off = air_struct_member_offset(e->m, t, i);
         else if (t->kind == AIR_TYPE_ARRAY || t->kind == AIR_TYPE_VECTOR)
            off = i * air_type_alloc_size(e->m, t->elem);
         else
            return false;
         if (!init_bytes(e, gi, offset + off, kc->elems[i]))
            return false;
      }
      return true;
   default:
      return false;
   }
}

/* -- values ---------------------------------------------------------------- */

static bool
ev_const(struct ev *e, const struct air_constant *k, struct ev_val *out)
{
   const struct air_type *t = k->base.type;
   memset(out, 0, sizeof(*out));
   if (t->kind == AIR_TYPE_POINTER) {
      out->is_ptr = true;
      return const_pointer(e, &k->base, &out->global, &out->offset);
   }
   const struct air_type *sc = air_type_scalar(t);
   if (sc->kind != AIR_TYPE_INT && !air_type_is_float(sc))
      return false;
   out->bits = air_type_bit_size(sc);
   out->n = t->kind == AIR_TYPE_VECTOR ? (unsigned)t->count : 1;
   if (out->n > 16)
      return false;
   switch (k->ckind) {
   case AIR_CONST_INT:
   case AIR_CONST_FLOAT:
      if (t->kind == AIR_TYPE_VECTOR)
         return false;
      out->lane[0] = k->u & mask_of(out->bits);
      return true;
   case AIR_CONST_NULL:
   case AIR_CONST_UNDEF:
   case AIR_CONST_POISON:
      return true;
   case AIR_CONST_DATA:
      for (unsigned i = 0; i < out->n; i++)
         out->lane[i] = k->data[i] & mask_of(out->bits);
      return true;
   case AIR_CONST_AGGREGATE:
      if (k->num_elems != out->n)
         return false;
      for (unsigned i = 0; i < out->n; i++) {
         const struct air_constant *el = air_value_as_constant(k->elems[i]);
         if (!el)
            return false;
         if (el->ckind == AIR_CONST_INT || el->ckind == AIR_CONST_FLOAT)
            out->lane[i] = el->u & mask_of(out->bits);
         else if (el->ckind != AIR_CONST_NULL && el->ckind != AIR_CONST_UNDEF &&
                  el->ckind != AIR_CONST_POISON)
            return false;
      }
      return true;
   default:
      return false;
   }
}

static bool
ev_get(struct ev *e, const struct air_value *v, struct ev_val *out)
{
   if (v->kind == AIR_VALUE_GLOBAL) {
      memset(out, 0, sizeof(*out));
      out->is_ptr = true;
      int gi = global_index(e, v);
      if (gi < 0)
         return false;
      out->global = (unsigned)gi;
      return true;
   }
   if (v->kind == AIR_VALUE_INSTRUCTION) {
      struct hash_entry *he = _mesa_hash_table_search(e->vals, v);
      if (!he)
         return false;
      *out = *(struct ev_val *)he->data;
      return true;
   }
   const struct air_constant *k = air_value_as_constant(v);
   return k && ev_const(e, k, out);
}

static void
ev_set(struct ev *e, const struct air_value *v, const struct ev_val *val)
{
   struct ev_val *copy = ralloc(e->c->mem_ctx, struct ev_val);
   *copy = *val;
   _mesa_hash_table_insert(e->vals, v, copy);
}

static int64_t
sext_of(uint64_t x, unsigned bits)
{
   if (bits >= 64)
      return (int64_t)x;
   uint64_t sign = 1ull << (bits - 1);
   x &= mask_of(bits);
   return (int64_t)((x ^ sign) - sign);
}

/* -- memory ------------------------------------------------------------------ */

static bool
ev_load(struct ev *e, const struct air_inst *in, const struct ev_val *p, struct ev_val *out)
{
   const struct air_type *t = in->base.type;
   const struct air_global *g = e->m->globals[p->global];
   struct air_known *k = e->k;

   memset(out, 0, sizeof(*out));
   if (t->kind == AIR_TYPE_POINTER) {
      for (unsigned i = 0; i < k->num_ptrs; i++) {
         if (k->ptrs[i].global == p->global && k->ptrs[i].offset == p->offset) {
            out->is_ptr = true;
            out->global = k->ptrs[i].target;
            return true;
         }
      }
      return false;
   }
   const struct air_type *sc = air_type_scalar(t);
   if (sc->kind != AIR_TYPE_INT && !air_type_is_float(sc))
      return false;
   out->bits = air_type_bit_size(sc);
   out->n = t->kind == AIR_TYPE_VECTOR ? (unsigned)t->count : 1;
   if (out->n > 16)
      return false;

   /* A function constant: the pipeline's value, its components at bit i * width
    * of the one word (air_constant_value), or zero when the pipeline gave none --
    * which is what an initializer guarded by air.is_function_constant_defined
    * reads and then discards.
    */
   int fc = function_constant_of(e->c, g);
   if (fc >= 0) {
      if (p->offset != 0)
         return false;
      const struct air_constant_value *cv = fc_value(e->c, (unsigned)fc);
      uint64_t word = cv ? cv->bits : 0;
      unsigned w = MAX2(out->bits, 8);
      for (unsigned i = 0; i < out->n; i++)
         out->lane[i] = (w * i < 64 ? word >> (w * i) : 0) & mask_of(out->bits);
      return true;
   }
   if (!k->bytes[p->global])
      return false;
   uint64_t stride = t->kind == AIR_TYPE_VECTOR ? MAX2(out->bits, 8) / 8 : 0;
   unsigned bytes = MAX2(out->bits, 8) / 8;
   for (unsigned i = 0; i < out->n; i++) {
      uint64_t at = p->offset + i * stride, x = 0;
      if (at + bytes > k->size[p->global])
         return false;
      for (unsigned b = 0; b < bytes; b++)
         x |= (uint64_t)k->bytes[p->global][at + b] << (8 * b);
      out->lane[i] = x & mask_of(out->bits);
   }
   return true;
}

static bool
ev_store(struct ev *e, const struct ev_val *p, const struct ev_val *v)
{
   struct air_known *k = e->k;
   if (!k->bytes[p->global])
      return false;
   if (v->is_ptr) {
      if (v->offset != 0)
         return false;
      add_pointer_cell(e, p->global, p->offset, v->global);
      return true;
   }
   unsigned bytes = MAX2(v->bits, 8) / 8;
   for (unsigned i = 0; i < v->n; i++) {
      if (p->offset + (i + 1) * bytes > k->size[p->global])
         return false;
      write_bytes(k->bytes[p->global], k->size[p->global], p->offset + i * bytes,
                  v->lane[i], bytes);
   }
   return true;
}

/* -- instructions ------------------------------------------------------------- */

static bool
ev_icmp(unsigned pred, uint64_t a, uint64_t b, unsigned bits)
{
   switch (pred) {
   case AIR_ICMP_EQ:  return a == b;
   case AIR_ICMP_NE:  return a != b;
   case AIR_ICMP_UGT: return a > b;
   case AIR_ICMP_UGE: return a >= b;
   case AIR_ICMP_ULT: return a < b;
   case AIR_ICMP_ULE: return a <= b;
   case AIR_ICMP_SGT: return sext_of(a, bits) > sext_of(b, bits);
   case AIR_ICMP_SGE: return sext_of(a, bits) >= sext_of(b, bits);
   case AIR_ICMP_SLT: return sext_of(a, bits) < sext_of(b, bits);
   case AIR_ICMP_SLE: return sext_of(a, bits) <= sext_of(b, bits);
   default:           return false;
   }
}

static bool
ev_binop(unsigned op, uint64_t a, uint64_t b, unsigned bits, uint64_t *out)
{
   uint64_t m = mask_of(bits);
   switch (op) {
   case AIR_BINOP_ADD:  *out = (a + b) & m; return true;
   case AIR_BINOP_SUB:  *out = (a - b) & m; return true;
   case AIR_BINOP_MUL:  *out = (a * b) & m; return true;
   case AIR_BINOP_AND:  *out = a & b; return true;
   case AIR_BINOP_OR:   *out = a | b; return true;
   case AIR_BINOP_XOR:  *out = (a ^ b) & m; return true;
   case AIR_BINOP_SHL:  *out = b < bits ? (a << b) & m : 0; return true;
   case AIR_BINOP_LSHR: *out = b < bits ? (a & m) >> b : 0; return true;
   case AIR_BINOP_ASHR:
      *out = b < bits ? (uint64_t)(sext_of(a, bits) >> b) & m : 0;
      return true;
   case AIR_BINOP_UDIV: if (!b) return false; *out = (a / b) & m; return true;
   case AIR_BINOP_UREM: if (!b) return false; *out = (a % b) & m; return true;
   case AIR_BINOP_SDIV:
      if (!b) return false;
      *out = (uint64_t)(sext_of(a, bits) / sext_of(b, bits)) & m;
      return true;
   case AIR_BINOP_SREM:
      if (!b) return false;
      *out = (uint64_t)(sext_of(a, bits) % sext_of(b, bits)) & m;
      return true;
   default:
      return false;
   }
}

static bool
ev_call(struct ev *e, const struct air_inst *in, struct ev_val *out)
{
   const struct air_value *callee = in->operands[0];
   const char *name = callee->kind == AIR_VALUE_FUNCTION ? callee->name : NULL;
   memset(out, 0, sizeof(*out));
   if (!name)
      return false;
   if (!strcmp(name, "air.is_function_constant_defined") && in->num_operands == 2) {
      /* the global itself, or (typed pointers) a constant cast of it */
      unsigned g;
      uint64_t off;
      if (!const_pointer(e, in->operands[1], &g, &off) || off != 0)
         return false;
      int fc = function_constant_of(e->c, e->m->globals[g]);
      if (fc < 0)
         return false;
      out->n = 1;
      out->bits = 1;
      out->lane[0] = fc_value(e->c, (unsigned)fc) != NULL;
      return true;
   }
   if (!strncmp(name, "air.normalize_function_constant_predicate.", 42) &&
       in->num_operands == 2) {
      struct ev_val a;
      if (!ev_get(e, in->operands[1], &a) || a.is_ptr || a.n != 1)
         return false;
      out->n = 1;
      out->bits = air_type_bit_size(in->base.type);
      out->lane[0] = a.lane[0] != 0;
      return true;
   }
   return false;
}

static bool
ev_inst(struct ev *e, const struct air_inst *in)
{
   struct ev_val r, a, b, cnd;
   memset(&r, 0, sizeof(r));

   switch (in->op) {
   case AIR_OP_LOAD:
      if (!ev_get(e, in->operands[0], &a) || !a.is_ptr || !ev_load(e, in, &a, &r))
         return false;
      break;
   case AIR_OP_STORE:
      return ev_get(e, in->operands[1], &a) && a.is_ptr &&
             ev_get(e, in->operands[0], &b) && ev_store(e, &a, &b);
   case AIR_OP_CMP:
      if (in->sub < AIR_ICMP_EQ || !ev_get(e, in->operands[0], &a) ||
          !ev_get(e, in->operands[1], &b) || a.is_ptr || b.is_ptr || a.n != b.n)
         return false;
      r.n = a.n;
      r.bits = 1;
      for (unsigned i = 0; i < a.n; i++)
         r.lane[i] = ev_icmp(in->sub, a.lane[i], b.lane[i], a.bits);
      break;
   case AIR_OP_BINOP:
      if (!ev_get(e, in->operands[0], &a) || !ev_get(e, in->operands[1], &b) ||
          a.is_ptr || b.is_ptr || a.n != b.n || air_type_is_float(air_type_scalar(in->base.type)))
         return false;
      r.n = a.n;
      r.bits = a.bits;
      for (unsigned i = 0; i < a.n; i++)
         if (!ev_binop(in->sub, a.lane[i], b.lane[i], a.bits, &r.lane[i]))
            return false;
      break;
   case AIR_OP_CAST: {
      if (!ev_get(e, in->operands[0], &a))
         return false;
      if (a.is_ptr) {
         if (in->sub != AIR_CAST_BITCAST && in->sub != AIR_CAST_ADDRSPACECAST)
            return false;
         r = a;
         break;
      }
      const struct air_type *rt = in->base.type;
      unsigned rbits = air_type_bit_size(air_type_scalar(rt));
      r.n = a.n;
      r.bits = rbits;
      for (unsigned i = 0; i < a.n; i++) {
         switch (in->sub) {
         case AIR_CAST_ZEXT:
         case AIR_CAST_TRUNC:
            r.lane[i] = a.lane[i] & mask_of(rbits);
            break;
         case AIR_CAST_SEXT:
            r.lane[i] = (uint64_t)sext_of(a.lane[i], a.bits) & mask_of(rbits);
            break;
         case AIR_CAST_BITCAST:
            if (rbits != a.bits || rt->kind == AIR_TYPE_POINTER)
               return false;
            r.lane[i] = a.lane[i];
            break;
         default:
            return false;
         }
      }
      break;
   }
   case AIR_OP_SELECT:
      if (!ev_get(e, in->operands[0], &cnd) || !ev_get(e, in->operands[1], &a) ||
          !ev_get(e, in->operands[2], &b) || cnd.is_ptr)
         return false;
      if (cnd.n == 1) {
         r = cnd.lane[0] & 1 ? a : b;
      } else {
         if (a.is_ptr || b.is_ptr || a.n != cnd.n || b.n != cnd.n)
            return false;
         r = a;
         for (unsigned i = 0; i < cnd.n; i++)
            r.lane[i] = cnd.lane[i] & 1 ? a.lane[i] : b.lane[i];
      }
      break;
   case AIR_OP_GEP: {
      uint64_t off;
      if (!ev_get(e, in->operands[0], &a) || !a.is_ptr ||
          !gep_offset(e, in->aux_type, in->operands + 1, in->num_operands - 1, NULL, &off))
         return false;
      r = a;
      r.offset += off;
      break;
   }
   case AIR_OP_EXTRACTELT: {
      uint64_t idx;
      if (!ev_get(e, in->operands[0], &a) || a.is_ptr ||
          !air_const_uint(in->operands[1], &idx) || idx >= a.n)
         return false;
      r.n = 1;
      r.bits = a.bits;
      r.lane[0] = a.lane[idx];
      break;
   }
   case AIR_OP_INSERTELT: {
      uint64_t idx;
      if (!ev_get(e, in->operands[0], &a) || !ev_get(e, in->operands[1], &b) || a.is_ptr ||
          b.is_ptr || !air_const_uint(in->operands[2], &idx) || idx >= a.n)
         return false;
      r = a;
      r.lane[idx] = b.lane[0];
      break;
   }
   case AIR_OP_FREEZE:
      if (!ev_get(e, in->operands[0], &r))
         return false;
      break;
   case AIR_OP_CALL:
      if (!ev_call(e, in, &r))
         return false;
      break;
   default:
      return false;
   }
   if (in->base.type->kind != AIR_TYPE_VOID)
      ev_set(e, &in->base, &r);
   return true;
}

/* AIR_EVAL_DEBUG=1 in the environment: say which instruction the evaluator
 * stopped at, for a module whose initializers are translated as code. */
static void
ev_say(const struct air_inst *in, const char *what)
{
   static int debug = -1;
   if (debug < 0)
      debug = getenv("AIR_EVAL_DEBUG") != NULL;
   if (debug)
      fprintf(stderr, "air_partial: %s at opcode %u sub %u (value %u, block %u%s%s)\n", what,
              in ? in->op : 0, in ? in->sub : 0, in ? in->base.id : 0,
              in && in->parent ? in->parent->index : 0,
              in && in->op == AIR_OP_CALL && in->operands[0]->name ? ", calling " : "",
              in && in->op == AIR_OP_CALL && in->operands[0]->name ? in->operands[0]->name : "");
}

/* One initializer, run from its entry block to its ret. */
static bool
ev_run(struct ev *e, const struct air_function *f)
{
   if (f->num_args || !f->num_blocks)
      return false;
   const struct air_block *blk = &f->blocks[0], *prev = NULL;
   for (unsigned steps = 0; steps < 1u << 20; steps++) {
      /* the phis first, all of them from the edge just taken */
      unsigned i = 0;
      struct ev_val phis[64];
      unsigned nphi = 0;
      for (; i < blk->num_insts && blk->insts[i]->op == AIR_OP_PHI; i++) {
         const struct air_inst *in = blk->insts[i];
         bool found = false;
         if (nphi == ARRAY_SIZE(phis))
            return false;
         for (unsigned k = 0; k < in->num_operands; k++) {
            if (in->targets[k] == prev) {
               if (!ev_get(e, in->operands[k], &phis[nphi]))
                  return false;
               found = true;
               break;
            }
         }
         if (!found)
            return false;
         nphi++;
      }
      for (unsigned p = 0; p < nphi; p++)
         ev_set(e, &blk->insts[p]->base, &phis[p]);

      for (; i + 1 < blk->num_insts; i++) {
         if (!ev_inst(e, blk->insts[i])) {
            ev_say(blk->insts[i], "an instruction it does not evaluate");
            return false;
         }
      }

      const struct air_inst *term = blk->insts[blk->num_insts - 1];
      const struct air_block *next = NULL;
      struct ev_val cnd;
      switch (term->op) {
      case AIR_OP_RET:
         return true;
      case AIR_OP_BR:
         if (term->num_targets == 1) {
            next = term->targets[0];
         } else {
            if (!ev_get(e, term->operands[0], &cnd) || cnd.is_ptr || cnd.n != 1)
               return false;
            next = cnd.lane[0] & 1 ? term->targets[0] : term->targets[1];
         }
         break;
      case AIR_OP_SWITCH: {
         if (!ev_get(e, term->operands[0], &cnd) || cnd.is_ptr || cnd.n != 1)
            return false;
         next = term->targets[0];
         for (unsigned k = 1; k < term->num_targets; k++) {
            struct ev_val v;
            if (!ev_get(e, term->operands[k], &v) || v.is_ptr)
               return false;
            if (v.lane[0] == cnd.lane[0]) {
               next = term->targets[k];
               break;
            }
         }
         break;
      }
      default:
         return false;
      }
      prev = blk;
      blk = next;
   }
   return false;
}

/* -- who else writes ------------------------------------------------------------- */

/* The global a value is derived from, or -1 for none and -2 for several. */
static int
taint_of(struct ev *e, struct hash_table *taint, const struct air_value *v)
{
   if (v->kind == AIR_VALUE_GLOBAL)
      return global_index(e, v);
   if (v->kind == AIR_VALUE_INSTRUCTION) {
      struct hash_entry *he = _mesa_hash_table_search(taint, v);
      return he ? (int)(intptr_t)he->data : -1;
   }
   const struct air_constant *k = air_value_as_constant(v);
   if (k && k->ckind == AIR_CONST_EXPR && k->num_operands &&
       (k->expr_op == AIR_OP_GEP || k->expr_op == AIR_OP_CAST))
      return taint_of(e, taint, k->operands[0]);
   return -1;
}

static int
taint_join(int a, int b)
{
   if (a == -1)
      return b;
   if (b == -1 || a == b)
      return a;
   return -2;
}

static void
mark_written(struct ev *e, int t)
{
   if (t >= 0) {
      e->k->written[t] = true;
   } else if (t == -2) {
      for (unsigned i = 0; i < e->m->num_globals; i++)
         e->k->written[i] = true;
   }
}

/* Every global the functions other than the initializers could write: through a
 * store, an atomic, a pointer stored to memory or handed to a function of the
 * module. Two passes over each function, which carries a derived pointer
 * through a phi of the loop it is in.
 */
static void
find_writers(struct ev *e)
{
   for (unsigned fi = 0; fi < e->m->num_functions; fi++) {
      const struct air_function *f = e->m->functions[fi];
      if (f->is_declaration || (f->section && !strcmp(f->section, "air.static_init")))
         continue;
      struct hash_table *taint = _mesa_pointer_hash_table_create(e->c->mem_ctx);
      for (unsigned pass = 0; pass < 2; pass++) {
         for (unsigned bi = 0; bi < f->num_blocks; bi++) {
            const struct air_block *blk = &f->blocks[bi];
            for (unsigned ii = 0; ii < blk->num_insts; ii++) {
               const struct air_inst *in = blk->insts[ii];
               int t = -1;
               switch (in->op) {
               case AIR_OP_GEP:
               case AIR_OP_CAST:
               case AIR_OP_FREEZE:
                  t = taint_of(e, taint, in->operands[0]);
                  break;
               case AIR_OP_SELECT:
                  t = taint_join(taint_of(e, taint, in->operands[1]),
                                 taint_of(e, taint, in->operands[2]));
                  break;
               case AIR_OP_PHI:
                  for (unsigned k = 0; k < in->num_operands; k++)
                     t = taint_join(t, taint_of(e, taint, in->operands[k]));
                  break;
               case AIR_OP_STORE:
                  mark_written(e, taint_of(e, taint, in->operands[1]));
                  mark_written(e, taint_of(e, taint, in->operands[0]));
                  break;
               case AIR_OP_ATOMICRMW:
               case AIR_OP_CMPXCHG:
                  mark_written(e, taint_of(e, taint, in->operands[0]));
                  break;
               case AIR_OP_CALL: {
                  const struct air_value *callee = in->operands[0];
                  bool intrinsic = callee->kind == AIR_VALUE_FUNCTION && callee->name &&
                                   (!strncmp(callee->name, "air.", 4) ||
                                    !strncmp(callee->name, "llvm.", 5));
                  for (unsigned k = 1; k < in->num_operands; k++) {
                     int at = taint_of(e, taint, in->operands[k]);
                     /* an intrinsic reads what it is given unless it is a
                      * store of some kind, which the corpus names as such */
                     if (!intrinsic ||
                         (callee->name && (strstr(callee->name, "write") ||
                                           strstr(callee->name, "atomic") ||
                                           strstr(callee->name, "memcpy") ||
                                           strstr(callee->name, "memset") ||
                                           strstr(callee->name, "memmove"))))
                        mark_written(e, at);
                  }
                  break;
               }
               default:
                  break;
               }
               if (t != -1 && in->base.type->kind != AIR_TYPE_VOID)
                  _mesa_hash_table_insert(taint, &in->base, (void *)(intptr_t)t);
            }
         }
      }
      _mesa_hash_table_destroy(taint, NULL);
   }
}

/* -- the entry ------------------------------------------------------------------ */

bool
air_eval_static_initializers(struct air_ctx *c)
{
   const struct air_module *m = c->m;
   struct air_known *k = &c->known;
   bool any = false;

   for (unsigned i = 0; i < m->num_functions; i++) {
      const struct air_function *f = m->functions[i];
      if (!f->is_declaration && f->section && !strcmp(f->section, "air.static_init"))
         any = true;
   }

   memset(k, 0, sizeof(*k));
   unsigned n = MAX2(m->num_globals, 1);
   k->bytes = rzalloc_array(c->mem_ctx, uint8_t *, n);
   k->size = rzalloc_array(c->mem_ctx, uint64_t, n);
   k->written = rzalloc_array(c->mem_ctx, bool, n);

   struct ev e = {
      .c = c,
      .m = m,
      .k = k,
      .index = _mesa_pointer_hash_table_create(c->mem_ctx),
      .vals = _mesa_pointer_hash_table_create(c->mem_ctx),
   };
   for (unsigned i = 0; i < m->num_globals; i++)
      _mesa_hash_table_insert(e.index, &m->globals[i]->base, (void *)(uintptr_t)(i + 1));

   /* every program-scope variable's bytes, as its initializer has them */
   for (unsigned i = 0; i < m->num_globals; i++) {
      const struct air_global *g = m->globals[i];
      if (!is_program_variable(g))
         continue;
      uint64_t size = air_type_alloc_size(m, g->value_type);
      if (!size || size > 1u << 20)
         continue;
      k->size[i] = size;
      k->bytes[i] = rzalloc_array(c->mem_ctx, uint8_t, size);
      if (g->init && !init_bytes(&e, i, 0, g->init)) {
         k->bytes[i] = NULL;
         k->size[i] = 0;
      }
   }

   /* the initializers, in the module's order, which is the order
    * run_static_initializers runs them in */
   for (unsigned i = 0; i < m->num_functions; i++) {
      const struct air_function *f = m->functions[i];
      if (f->is_declaration || !f->section || strcmp(f->section, "air.static_init"))
         continue;
      if (!ev_run(&e, f)) {
         memset(k, 0, sizeof(*k));
         return false;
      }
   }
   find_writers(&e);
   k->evaluated = true;
   k->had_initializers = any;
   return true;
}
