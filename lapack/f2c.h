/* f2c.h  --  Standard Fortran to C header file */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

#ifndef F2C_INCLUDE
#define F2C_INCLUDE

#include <math.h>
#if defined(__LAPACK_PRECISION_QUAD)
#	include <quadmath.h>
#	define M(A) A##q
	typedef __float128 quadreal;
	typedef struct { quadreal r, i; } quadcomplex;
#	define scalar __float128
#	define scalarcomplex quadcomplex
#	define dscalar __float128
#elif defined(__LAPACK_PRECISION_HALF)
#	define M(A) A##f
	typedef __fp16 halfreal;
	typedef struct { halfreal r, i; } halfcomplex;
#	define scalar __fp16
#	define scalarcomplex halfcomplex
#	define dscalar __fp16
#elif defined( __LAPACK_PRECISION_SINGLE)
#	define M(A) A##f
#	define scalar float
#	define scalarcomplex complex
#	define dscalar double
#else
#	define M(A) A
#	define scalar double
#	define scalarcomplex doublecomplex
#	define dscalar double
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef int integer;
typedef unsigned int uinteger;
typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;
typedef struct { real r, i; } complex;
typedef struct { doublereal r, i; } doublecomplex;
typedef int logical;
typedef short int shortlogical;
typedef char logical1;
typedef char integer1;

#define TRUE_ (1)
#define FALSE_ (0)

/* Extern is for use with -E */
#ifndef Extern
#define Extern extern
#endif

/* I/O stuff */

typedef int flag;
typedef int ftnlen;
typedef int ftnint;

/*external read, write*/
typedef struct
{	flag cierr;
	ftnint ciunit;
	flag ciend;
	char *cifmt;
	ftnint cirec;
} cilist;

/*internal read, write*/
typedef struct
{	flag icierr;
	char *iciunit;
	flag iciend;
	char *icifmt;
	ftnint icirlen;
	ftnint icirnum;
} icilist;

/*open*/
typedef struct
{	flag oerr;
	ftnint ounit;
	char *ofnm;
	ftnlen ofnmlen;
	char *osta;
	char *oacc;
	char *ofm;
	ftnint orl;
	char *oblnk;
} olist;

/*close*/
typedef struct
{	flag cerr;
	ftnint cunit;
	char *csta;
} cllist;

/*rewind, backspace, endfile*/
typedef struct
{	flag aerr;
	ftnint aunit;
} alist;

/* inquire */
typedef struct
{	flag inerr;
	ftnint inunit;
	char *infile;
	ftnlen infilen;
	ftnint	*inex;	/*parameters in standard's order*/
	ftnint	*inopen;
	ftnint	*innum;
	ftnint	*innamed;
	char	*inname;
	ftnlen	innamlen;
	char	*inacc;
	ftnlen	inacclen;
	char	*inseq;
	ftnlen	inseqlen;
	char 	*indir;
	ftnlen	indirlen;
	char	*infmt;
	ftnlen	infmtlen;
	char	*inform;
	ftnint	informlen;
	char	*inunf;
	ftnlen	inunflen;
	ftnint	*inrecl;
	ftnint	*innrec;
	char	*inblank;
	ftnlen	inblanklen;
} inlist;

#define VOID void

union Multitype {	/* for multiple entry points */
	integer1 g;
	shortint h;
	integer i;
	/* longint j; */
	real r;
	doublereal d;
	complex c;
	doublecomplex z;
	};

typedef union Multitype Multitype;

struct Vardesc {	/* for Namelist */
	char *name;
	char *addr;
	ftnlen *dims;
	int  type;
	};
typedef struct Vardesc Vardesc;

struct Namelist {
	char *name;
	Vardesc **vars;
	int nvars;
	};
typedef struct Namelist Namelist;

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) (abs(x))
#define f2cmin(a,b) ((a) <= (b) ? (a) : (b))
#define f2cmax(a,b) ((a) >= (b) ? (a) : (b))
#define dmin(a,b) (f2cmin(a,b))
#define dmax(a,b) (f2cmax(a,b))
#define bit_test(a,b)	((a) >> (b) & 1)
#define bit_clear(a,b)	((a) & ~((uinteger)1 << (b)))
#define bit_set(a,b)	((a) |  ((uinteger)1 << (b)))

#define abort_() { sig_die("Fortran abort routine called", 1); }
#if defined(__LAPACK_PRECISION_QUAD)
#	define f__cabs(r,i) qf__cabs((r),(i))
	extern scalar qf__cabs(scalar r, scalar i);
#elif defined(__LAPACK_PRECISION_HALF)
#	define f__cabs(r,i) hf__cabs((r),(i))
	extern scalar hf__cabs(scalar r, scalar i);
#elif defined( __LAPACK_PRECISION_SINGLE)
#	define f__cabs(r,i) sf__cabs((r),(i))
	extern scalar sf__cabs(scalar r, scalar i);
#else
#	define f__cabs(r,i) df__cabs((r),(i))
	extern scalar df__cabs(scalar r, scalar i);
#endif
#define c_abs(z) ( f__cabs( (z)->r, (z)->i ) )
#define c_cos(R,Z) {(R)->r = (M(cos)((Z)->r) * M(cosh)((Z)->i)); (R)->i = (-M(sin)((Z)->r) * M(sinh)((Z)->i));}
#define c_div(c, a, b) { 	scalar __ratio, __den, __abr, __abi, __cr; 	if( (__abr = (b)->r) < 0.) 		__abr = - __abr; 	if( (__abi = (b)->i) < 0.) 		__abi = - __abi; 	if( __abr <= __abi ) 		{ 		if(__abi == 0) 			sig_die("complex division by zero", 1); 		__ratio = (b)->r / (b)->i ; 		__den = (b)->i * (M(1.0) + __ratio*__ratio); 		__cr = ((a)->r*__ratio + (a)->i) / __den; 		(c)->i = ((a)->i*__ratio - (a)->r) / __den; 		} 	else 		{ 		__ratio = (b)->i / (b)->r ; 		__den = (b)->r * (1 + __ratio*__ratio); 		__cr = ((a)->r + (a)->i*__ratio) / __den; 		(c)->i = ((a)->i - (a)->r*__ratio) / __den; 		} 	(c)->r = __cr; 	}
#define z_div(c, a, b) c_div(c, a, b)
#define c_exp(R, Z) { (R)->r = (M(exp)((Z)->r) * M(cos)((Z)->i)); (R)->i = (M(exp)((Z)->r) * M(sin)((Z)->i)); }
#define c_log(R, Z) { (R)->i = M(atan2)((Z)->i, (Z)->r); (R)->r = M(log)( f__cabs((Z)->r, (Z)->i) ); }
#define c_sin(R, Z) { (R)->r = (M(sin)((Z)->r) * M(cosh)((Z)->i)); (R)->i = (M(cos)((Z)->r) * M(sinh)((Z)->i)); }
#define c_sqrt(R, Z) { 	scalar __mag, __t, __zi = (Z)->i, __zr = (Z)->r; 	if( (__mag = f__cabs(__zr, __zi)) == 0.) (R)->r = (R)->i = 0.; 	else if(__zr > 0) { 		(R)->r = __t = M(sqrt)(M(0.5) * (__mag + __zr) ); 		__t = __zi / __t; 		(R)->i = M(0.5) * __t; 	} else { 		__t = M(sqrt)(M(0.5) * (__mag - __zr) ); 		if(__zi < 0) __t = -__t; 		(R)->i = __t; 		__t = __zi / __t; 		(R)->r = M(0.5) * __t; 	} }
#define d_abs(x) abs(*(x))
#define d_acos(x) (M(acos)(*(x)))
#define d_asin(x) (M(asin)(*(x)))
#define d_atan(x) (M(atan)(*(x)))
#define d_atn2(x, y) (M(atan2)(*(x),*(y)))
#define d_cnjg(R, Z) { (R)->r = (Z)->r;	(R)->i = -((Z)->i); }
#define d_cos(x) (M(cos)(*(x)))
#define d_cosh(x) (M(cosh)(*(x)))
#define d_dim(__a, __b) ( *(__a) > *(__b) ? *(__a) - *(__b) : 0.0 )
#define d_exp(x) (M(exp)(*(x)))
#define d_imag(z) ((z)->i)
#define d_int(__x) (*(__x)>0 ? M(floor)(*(__x)) : -M(floor)(- *(__x)))
#define d_lg10(x) ( M(0.43429448190325182765) * M(log)(*(x)) )
#define d_log(x) (M(log)(*(x)))
#define d_mod(x, y) (M(fmod)(*(x), *(y)))
#define u_nint(__x) ((__x)>=0 ? M(floor)((__x) + M(.5)) : -M(floor)(M(.5) - (__x)))
#define d_nint(x) u_nint(*(x))
#define u_sign(__a,__b) ((__b) >= 0 ? ((__a) >= 0 ? (__a) : -(__a)) : -((__a) >= 0 ? (__a) : -(__a)))
#define d_sign(a,b) u_sign(*(a),*(b))
#define d_sin(x) (M(sin)(*(x)))
#define d_sinh(x) (M(sinh)(*(x)))
#define d_sqrt(x) (M(sqrt)(*(x)))
#define d_tan(x) (M(tan)(*(x)))
#define d_tanh(x) (M(tanh)(*(x)))
#define i_abs(x) abs(*(x))
#define i_dnnt(x) ((integer)u_nint(*(x)))
#define i_len(s, n) (n)
#define i_nint(x) ((integer)u_nint(*(x)))
#define i_sign(a,b) ((integer)u_sign((integer)*(a),(integer)*(b)))
#define pow_ci(p, a, b) { pow_zi((p), (a), (b)); }
#define pow_dd(ap, bp) (M(pow)(*(ap), *(bp)))
#if defined(__LAPACK_PRECISION_QUAD)
#	define pow_di(B,E) qpow_ui((B),*(E))
	extern dscalar qpow_ui(scalar *_x, integer n);
#elif defined(__LAPACK_PRECISION_HALF)
#	define pow_di(B,E) hpow_ui((B),*(E))
	extern dscalar hpow_ui(scalar *_x, integer n);
#elif defined( __LAPACK_PRECISION_SINGLE)
#	define pow_ri(B,E) spow_ui((B),*(E))
	extern dscalar spow_ui(scalar *_x, integer n);
#else
#	define pow_di(B,E) dpow_ui((B),*(E))
	extern dscalar dpow_ui(scalar *_x, integer n);
#endif
extern integer pow_ii(integer*,integer*);
#define pow_zi(p, a, b) { 	integer __n=*(b); unsigned long __u; scalar __t; scalarcomplex __x; 	static scalarcomplex one = {1.0, 0.0}; 	(p)->r = 1; (p)->i = 0; 	if(__n != 0) { 		if(__n < 0) { 			__n = -__n; 			z_div(&__x, &one, (a)); 		} else { 			__x.r = (a)->r; __x.i = (a)->i; 		} 		for(__u = __n; ; ) { 			if(__u & 01) { 				__t = (p)->r * __x.r - (p)->i * __x.i; 				(p)->i = (p)->r * __x.i + (p)->i * __x.r; 				(p)->r = __t; 			} 			if(__u >>= 1) { 				__t = __x.r * __x.r - __x.i * __x.i; 				__x.i = 2 * __x.r * __x.i; 				__x.r = __t; 			} else break; 		} 	} }
#define pow_zz(R,A,B) { 	scalar __logr, __logi, __x, __y; 	__logr = M(log)( f__cabs((A)->r, (A)->i) ); 	__logi = M(atan2)((A)->i, (A)->r); 	__x = M(exp)( __logr * (B)->r - __logi * (B)->i ); 	__y = __logr * (B)->i + __logi * (B)->r; 	(R)->r = __x * M(cos)(__y); 	(R)->i = __x * M(sin)(__y); }
#define r_cnjg(R, Z) d_cnjg(R,Z)
#define r_imag(z) d_imag(z)
#define r_lg10(x) d_lg10(x)
#define r_sign(a,b) d_sign(a,b)
#define s_cat(lpp, rpp, rnp, np, llp) { 	ftnlen i, nc, ll; char *f__rp, *lp; 	ll = (llp); lp = (lpp); 	for(i=0; i < (int)*(np); ++i) {         	nc = ll; 	        if((rnp)[i] < nc) nc = (rnp)[i]; 	        ll -= nc;         	f__rp = (rpp)[i]; 	        while(--nc >= 0) *lp++ = *(f__rp)++;         } 	while(--ll >= 0) *lp++ = ' '; }
#define s_cmp(a,b,c,d) ((integer)strncmp((a),(b),f2cmin((c),(d))))
#define s_copy(A,B,C,D) { strncpy((A),(B),f2cmin((C),(D))); }
#define sig_die(s, kill) { exit(1); }
#define s_stop(s, n) {exit(0);}
static char junk[] = "\n@(#)LIBF77 VERSION 19990503\n";
#define z_abs(z) c_abs(z)
#define z_exp(R, Z) c_exp(R, Z)
#define z_sqrt(R, Z) c_sqrt(R, Z)
#define myexit_() break;
#if defined(__LAPACK_PRECISION_QUAD)
#	define mymaxloc_(w,s,e,n) qmaxloc_((w),*(s),*(e),n)
	extern integer qmaxloc_(scalar *w, integer s, integer e, integer *n);
#elif defined(__LAPACK_PRECISION_HALF)
#	define mymaxloc_(w,s,e,n) hmaxloc_((w),*(s),*(e),n)
	extern integer hmaxloc_(scalar *w, integer s, integer e, integer *n);
#elif defined( __LAPACK_PRECISION_SINGLE)
#	define mymaxloc_(w,s,e,n) smaxloc_((w),*(s),*(e),n)
	extern integer smaxloc_(scalar *w, integer s, integer e, integer *n);
#else
#	define mymaxloc_(w,s,e,n) dmaxloc_((w),*(s),*(e),n)
	extern integer dmaxloc_(scalar *w, integer s, integer e, integer *n);
#endif

/* procedure parameter types for -A and -C++ */

#define F2C_proc_par_types 1
#ifdef __cplusplus
typedef logical (*L_fp)(...);
#else
typedef logical (*L_fp)();
#endif
#endif
