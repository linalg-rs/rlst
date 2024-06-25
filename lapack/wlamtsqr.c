#define len_trim__(cad,len) ({                           integer _r=0,i;                           for(i=0; i<(len) && (cad)[i]; i++)                             if((cad)[i] != ' ') _r=i;                           _r+1; })
#define ceiling_(a) (myceil(*(a)))
#define myceil(a) (sizeof(a) == sizeof(float) ? ceilf(a) : ceil(a))
#include <math.h>
/*  -- translated by f2c (version 20200916).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#define __LAPACK_PRECISION_QUAD
#include "f2c.h"

/* Table of constant values */

static integer c__0 = 0;


/*  Definition: */
/*  =========== */

/*      SUBROUTINE ZLAMTSQR( SIDE, TRANS, M, N, K, MB, NB, A, LDA, T, */
/*     $                     LDT, C, LDC, WORK, LWORK, INFO ) */


/*      CHARACTER         SIDE, TRANS */
/*      INTEGER           INFO, LDA, M, N, K, MB, NB, LDT, LWORK, LDC */
/*      COMPLEX*16        A( LDA, * ), WORK( * ), C(LDC, * ), */
/*     $                  T( LDT, * ) */
/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* >      ZLAMTSQR overwrites the general complex M-by-N matrix C with */
/* > */
/* > */
/* >                 SIDE = 'L'     SIDE = 'R' */
/* > TRANS = 'N':      Q * C          C * Q */
/* > TRANS = 'C':      Q**H * C       C * Q**H */
/* >      where Q is a doublereal orthogonal matrix defined as the product */
/* >      of blocked elementary reflectors computed by tall skinny */
/* >      QR factorization (ZLATSQR) */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] SIDE */
/* > \verbatim */
/* >          SIDE is CHARACTER*1 */
/* >          = 'L': apply Q or Q**H from the Left; */
/* >          = 'R': apply Q or Q**H from the Right. */
/* > \endverbatim */
/* > */
/* > \param[in] TRANS */
/* > \verbatim */
/* >          TRANS is CHARACTER*1 */
/* >          = 'N':  No transpose, apply Q; */
/* >          = 'C':  Conjugate Transpose, apply Q**H. */
/* > \endverbatim */
/* > */
/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix A.  M >=0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix C. M >= N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] K */
/* > \verbatim */
/* >          K is INTEGER */
/* >          The number of elementary reflectors whose product defines */
/* >          the matrix Q. */
/* >          N >= K >= 0; */
/* > */
/* > \endverbatim */
/* > */
/* > \param[in] MB */
/* > \verbatim */
/* >          MB is INTEGER */
/* >          The block size to be used in the blocked QR. */
/* >          MB > N. (must be the same as DLATSQR) */
/* > \endverbatim */
/* > */
/* > \param[in] NB */
/* > \verbatim */
/* >          NB is INTEGER */
/* >          The column block size to be used in the blocked QR. */
/* >          N >= NB >= 1. */
/* > \endverbatim */
/* > */
/* > \param[in] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,K) */
/* >          The i-th column must contain the vector which defines the */
/* >          blockedelementary reflector H(i), for i = 1,2,...,k, as */
/* >          returned by DLATSQR in the first k columns of */
/* >          its array argument A. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A. */
/* >          If SIDE = 'L', LDA >= f2cmax(1,M); */
/* >          if SIDE = 'R', LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] T */
/* > \verbatim */
/* >          T is COMPLEX*16 array, dimension */
/* >          ( N * Number of blocks(CEIL(M-K/MB-K)), */
/* >          The blocked upper triangular block reflectors stored in compact form */
/* >          as a sequence of upper triangular blocks.  See below */
/* >          for further details. */
/* > \endverbatim */
/* > */
/* > \param[in] LDT */
/* > \verbatim */
/* >          LDT is INTEGER */
/* >          The leading dimension of the array T.  LDT >= NB. */
/* > \endverbatim */
/* > */
/* > \param[in,out] C */
/* > \verbatim */
/* >          C is COMPLEX*16 array, dimension (LDC,N) */
/* >          On entry, the M-by-N matrix C. */
/* >          On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q. */
/* > \endverbatim */
/* > */
/* > \param[in] LDC */
/* > \verbatim */
/* >          LDC is INTEGER */
/* >          The leading dimension of the array C. LDC >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >         (workspace) COMPLEX*16 array, dimension (MAX(1,LWORK)) */
/* > */
/* > \endverbatim */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          The dimension of the array WORK. */
/* > */
/* >          If SIDE = 'L', LWORK >= f2cmax(1,N)*NB; */
/* >          if SIDE = 'R', LWORK >= f2cmax(1,MB)*NB. */
/* >          If LWORK = -1, then a workspace query is assumed; the routine */
/* >          only calculates the optimal size of the WORK array, returns */
/* >          this value as the first entry of the WORK array, and no error */
/* >          message related to LWORK is issued by XERBLA. */
/* > */
/* > \endverbatim */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > Tall-Skinny QR (TSQR) performs QR by a sequence of orthogonal transformations, */
/* > representing Q as a product of other orthogonal matrices */
/* >   Q = Q(1) * Q(2) * . . . * Q(k) */
/* > where each Q(i) zeros out subdiagonal entries of a block of MB rows of A: */
/* >   Q(1) zeros out the subdiagonal entries of rows 1:MB of A */
/* >   Q(2) zeros out the bottom MB-N rows of rows [1:N,MB+1:2*MB-N] of A */
/* >   Q(3) zeros out the bottom MB-N rows of rows [1:N,2*MB-N+1:3*MB-2*N] of A */
/* >   . . . */
/* > */
/* > Q(1) is computed by GEQRT, which represents Q(1) by Householder vectors */
/* > stored under the diagonal of rows 1:MB of A, and by upper triangular */
/* > block reflectors, stored in array T(1:LDT,1:N). */
/* > For more information see Further Details in GEQRT. */
/* > */
/* > Q(i) for i>1 is computed by TPQRT, which represents Q(i) by Householder vectors */
/* > stored in rows [(i-1)*(MB-N)+N+1:i*(MB-N)+N] of A, and by upper triangular */
/* > block reflectors, stored in array T(1:LDT,(i-1)*N+1:i*N). */
/* > The last Q(k) may use fewer rows. */
/* > For more information see Further Details in TPQRT. */
/* > */
/* > For more details of the overall algorithm, see the description of */
/* > Sequential TSQR in Section 2.2 of [1]. */
/* > */
/* > [1] “Communication-Optimal Parallel and Sequential QR and LU Factorizations, */
/* >     J. Demmel, L. Grigori, M. Hoemmen, J. Langou, */
/* >     SIAM J. Sci. Comput, vol. 34, no. 1, 2012 */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  wlamtsqr_(char *side, char *trans, integer *m, integer *
	n, integer *k, integer *mb, integer *nb, quadcomplex *a, integer *
	lda, quadcomplex *t, integer *ldt, quadcomplex *c__, integer *ldc,
	 quadcomplex *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, t_dim1, t_offset, i__1, i__2, 
	    i__3;

    /* Local variables */
    extern void  wtpmqrt_(char *, char *, integer *, integer *,
	     integer *, integer *, integer *, quadcomplex *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, integer *);
    integer i__, ii, kk, lw, ctr;
    logical left, tran;
    extern logical lsame_(char *, char *);
    logical right;
    extern void  xerbla_(char *, integer *);
    logical notran, lquery;
    extern void  wgemqrt_(char *, char *, integer *, integer *,
	     integer *, integer *, quadcomplex *, integer *, quadcomplex *
	    , integer *, quadcomplex *, integer *, quadcomplex *, integer 
	    *);


/*  -- LAPACK computational routine (version 3.7.1) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2017 */


/* ===================================================================== */


/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    lquery = *lwork < 0;
    notran = lsame_(trans, "N");
    tran = lsame_(trans, "C");
    left = lsame_(side, "L");
    right = lsame_(side, "R");
    if (left) {
	lw = *n * *nb;
    } else {
	lw = *m * *nb;
    }

    *info = 0;
    if (! left && ! right) {
	*info = -1;
    } else if (! tran && ! notran) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0) {
	*info = -5;
    } else if (*lda < f2cmax(1,*k)) {
	*info = -9;
    } else if (*ldt < f2cmax(1,*nb)) {
	*info = -11;
    } else if (*ldc < f2cmax(1,*m)) {
	*info = -13;
    } else if (*lwork < f2cmax(1,lw) && ! lquery) {
	*info = -15;
    }

/*     Determine the block size if it is tall skinny or short and wide */

    if (*info == 0) {
	work[1].r = (quadreal) lw, work[1].i = 0.;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZLAMTSQR", &i__1);
	return;
    } else if (lquery) {
	return;
    }

/*     Quick return if possible */

/* Computing MIN */
    i__1 = f2cmin(*m,*n);
    if (f2cmin(i__1,*k) == 0) {
	return;
    }

/* Computing MAX */
    i__1 = f2cmax(*m,*n);
    if (*mb <= *k || *mb >= f2cmax(i__1,*k)) {
	wgemqrt_(side, trans, m, n, k, nb, &a[a_offset], lda, &t[t_offset], 
		ldt, &c__[c_offset], ldc, &work[1], info);
	return;
    }

    if (left && notran) {

/*         Multiply Q to the last block of C */

	kk = (*m - *k) % (*mb - *k);
	ctr = (*m - *k) / (*mb - *k);
	if (kk > 0) {
	    ii = *m - kk + 1;
	    wtpmqrt_("L", "N", &kk, n, k, &c__0, nb, &a[ii + a_dim1], lda, &t[
		    (ctr * *k + 1) * t_dim1 + 1], ldt, &c__[c_dim1 + 1], ldc, 
		    &c__[ii + c_dim1], ldc, &work[1], info);
	} else {
	    ii = *m + 1;
	}

	i__1 = *mb + 1;
	i__2 = -(*mb - *k);
	for (i__ = ii - (*mb - *k); i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ 
		+= i__2) {

/*         Multiply Q to the current block of C (I:I+MB,1:N) */

	    --ctr;
	    i__3 = *mb - *k;
	    wtpmqrt_("L", "N", &i__3, n, k, &c__0, nb, &a[i__ + a_dim1], lda, 
		    &t[(ctr * *k + 1) * t_dim1 + 1], ldt, &c__[c_dim1 + 1], 
		    ldc, &c__[i__ + c_dim1], ldc, &work[1], info);
	}

/*         Multiply Q to the first block of C (1:MB,1:N) */

	wgemqrt_("L", "N", mb, n, k, nb, &a[a_dim1 + 1], lda, &t[t_offset], 
		ldt, &c__[c_dim1 + 1], ldc, &work[1], info);

    } else if (left && tran) {

/*         Multiply Q to the first block of C */

	kk = (*m - *k) % (*mb - *k);
	ii = *m - kk + 1;
	ctr = 1;
	wgemqrt_("L", "C", mb, n, k, nb, &a[a_dim1 + 1], lda, &t[t_offset], 
		ldt, &c__[c_dim1 + 1], ldc, &work[1], info);

	i__2 = ii - *mb + *k;
	i__1 = *mb - *k;
	for (i__ = *mb + 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1)
		 {

/*         Multiply Q to the current block of C (I:I+MB,1:N) */

	    i__3 = *mb - *k;
	    wtpmqrt_("L", "C", &i__3, n, k, &c__0, nb, &a[i__ + a_dim1], lda, 
		    &t[(ctr * *k + 1) * t_dim1 + 1], ldt, &c__[c_dim1 + 1], 
		    ldc, &c__[i__ + c_dim1], ldc, &work[1], info);
	    ++ctr;

	}
	if (ii <= *m) {

/*         Multiply Q to the last block of C */

	    wtpmqrt_("L", "C", &kk, n, k, &c__0, nb, &a[ii + a_dim1], lda, &t[
		    (ctr * *k + 1) * t_dim1 + 1], ldt, &c__[c_dim1 + 1], ldc, 
		    &c__[ii + c_dim1], ldc, &work[1], info);

	}

    } else if (right && tran) {

/*         Multiply Q to the last block of C */

	kk = (*n - *k) % (*mb - *k);
	ctr = (*n - *k) / (*mb - *k);
	if (kk > 0) {
	    ii = *n - kk + 1;
	    wtpmqrt_("R", "C", m, &kk, k, &c__0, nb, &a[ii + a_dim1], lda, &t[
		    (ctr * *k + 1) * t_dim1 + 1], ldt, &c__[c_dim1 + 1], ldc, 
		    &c__[ii * c_dim1 + 1], ldc, &work[1], info);
	} else {
	    ii = *n + 1;
	}

	i__1 = *mb + 1;
	i__2 = -(*mb - *k);
	for (i__ = ii - (*mb - *k); i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ 
		+= i__2) {

/*         Multiply Q to the current block of C (1:M,I:I+MB) */

	    --ctr;
	    i__3 = *mb - *k;
	    wtpmqrt_("R", "C", m, &i__3, k, &c__0, nb, &a[i__ + a_dim1], lda, 
		    &t[(ctr * *k + 1) * t_dim1 + 1], ldt, &c__[c_dim1 + 1], 
		    ldc, &c__[i__ * c_dim1 + 1], ldc, &work[1], info);
	}

/*         Multiply Q to the first block of C (1:M,1:MB) */

	wgemqrt_("R", "C", m, mb, k, nb, &a[a_dim1 + 1], lda, &t[t_offset], 
		ldt, &c__[c_dim1 + 1], ldc, &work[1], info);

    } else if (right && notran) {

/*         Multiply Q to the first block of C */

	kk = (*n - *k) % (*mb - *k);
	ii = *n - kk + 1;
	ctr = 1;
	wgemqrt_("R", "N", m, mb, k, nb, &a[a_dim1 + 1], lda, &t[t_offset], 
		ldt, &c__[c_dim1 + 1], ldc, &work[1], info);

	i__2 = ii - *mb + *k;
	i__1 = *mb - *k;
	for (i__ = *mb + 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1)
		 {

/*         Multiply Q to the current block of C (1:M,I:I+MB) */

	    i__3 = *mb - *k;
	    wtpmqrt_("R", "N", m, &i__3, k, &c__0, nb, &a[i__ + a_dim1], lda, 
		    &t[(ctr * *k + 1) * t_dim1 + 1], ldt, &c__[c_dim1 + 1], 
		    ldc, &c__[i__ * c_dim1 + 1], ldc, &work[1], info);
	    ++ctr;

	}
	if (ii <= *n) {

/*         Multiply Q to the last block of C */

	    wtpmqrt_("R", "N", m, &kk, k, &c__0, nb, &a[ii + a_dim1], lda, &t[
		    (ctr * *k + 1) * t_dim1 + 1], ldt, &c__[c_dim1 + 1], ldc, 
		    &c__[ii * c_dim1 + 1], ldc, &work[1], info);

	}

    }

    work[1].r = (quadreal) lw, work[1].i = 0.;
    return;

/*     End of ZLAMTSQR */

} /* wlamtsqr_ */

