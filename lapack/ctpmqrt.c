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

#define __LAPACK_PRECISION_SINGLE
#include "f2c.h"

/* > \brief \b CTPMQRT */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CTPMQRT + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ctpmqrt
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ctpmqrt
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ctpmqrt
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CTPMQRT( SIDE, TRANS, M, N, K, L, NB, V, LDV, T, LDT, */
/*                           A, LDA, B, LDB, WORK, INFO ) */

/*       CHARACTER SIDE, TRANS */
/*       INTEGER   INFO, K, LDV, LDA, LDB, M, N, L, NB, LDT */
/*       COMPLEX   V( LDV, * ), A( LDA, * ), B( LDB, * ), T( LDT, * ), */
/*      $          WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CTPMQRT applies a complex orthogonal matrix Q obtained from a */
/* > "triangular-pentagonal" complex block reflector H to a general */
/* > complex matrix C, which consists of two blocks A and B. */
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
/* >          = 'C':  Transpose, apply Q**H. */
/* > \endverbatim */
/* > */
/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix B. M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix B. N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] K */
/* > \verbatim */
/* >          K is INTEGER */
/* >          The number of elementary reflectors whose product defines */
/* >          the matrix Q. */
/* > \endverbatim */
/* > */
/* > \param[in] L */
/* > \verbatim */
/* >          L is INTEGER */
/* >          The order of the trapezoidal part of V. */
/* >          K >= L >= 0.  See Further Details. */
/* > \endverbatim */
/* > */
/* > \param[in] NB */
/* > \verbatim */
/* >          NB is INTEGER */
/* >          The block size used for the storage of T.  K >= NB >= 1. */
/* >          This must be the same value of NB used to generate T */
/* >          in CTPQRT. */
/* > \endverbatim */
/* > */
/* > \param[in] V */
/* > \verbatim */
/* >          V is COMPLEX array, dimension (LDA,K) */
/* >          The i-th column must contain the vector which defines the */
/* >          elementary reflector H(i), for i = 1,2,...,k, as returned by */
/* >          CTPQRT in B.  See Further Details. */
/* > \endverbatim */
/* > */
/* > \param[in] LDV */
/* > \verbatim */
/* >          LDV is INTEGER */
/* >          The leading dimension of the array V. */
/* >          If SIDE = 'L', LDV >= f2cmax(1,M); */
/* >          if SIDE = 'R', LDV >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] T */
/* > \verbatim */
/* >          T is COMPLEX array, dimension (LDT,K) */
/* >          The upper triangular factors of the block reflectors */
/* >          as returned by CTPQRT, stored as a NB-by-K matrix. */
/* > \endverbatim */
/* > */
/* > \param[in] LDT */
/* > \verbatim */
/* >          LDT is INTEGER */
/* >          The leading dimension of the array T.  LDT >= NB. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX array, dimension */
/* >          (LDA,N) if SIDE = 'L' or */
/* >          (LDA,K) if SIDE = 'R' */
/* >          On entry, the K-by-N or M-by-K matrix A. */
/* >          On exit, A is overwritten by the corresponding block of */
/* >          Q*C or Q**H*C or C*Q or C*Q**H.  See Further Details. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A. */
/* >          If SIDE = 'L', LDC >= f2cmax(1,K); */
/* >          If SIDE = 'R', LDC >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is COMPLEX array, dimension (LDB,N) */
/* >          On entry, the M-by-N matrix B. */
/* >          On exit, B is overwritten by the corresponding block of */
/* >          Q*C or Q**H*C or C*Q or C*Q**H.  See Further Details. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B. */
/* >          LDB >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX array. The dimension of WORK is */
/* >           N*NB if SIDE = 'L', or  M*NB if SIDE = 'R'. */
/* > \endverbatim */
/* > */
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

/* > \date November 2017 */

/* > \ingroup complexOTHERcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  The columns of the pentagonal matrix V contain the elementary reflectors */
/* >  H(1), H(2), ..., H(K); V is composed of a rectangular block V1 and a */
/* >  trapezoidal block V2: */
/* > */
/* >        V = [V1] */
/* >            [V2]. */
/* > */
/* >  The size of the trapezoidal block V2 is determined by the parameter L, */
/* >  where 0 <= L <= K; V2 is upper trapezoidal, consisting of the first L */
/* >  rows of a K-by-K upper triangular matrix.  If L=K, V2 is upper triangular; */
/* >  if L=0, there is no trapezoidal block, hence V = V1 is rectangular. */
/* > */
/* >  If SIDE = 'L':  C = [A]  where A is K-by-N,  B is M-by-N and V is M-by-K. */
/* >                      [B] */
/* > */
/* >  If SIDE = 'R':  C = [A B]  where A is M-by-K, B is M-by-N and V is N-by-K. */
/* > */
/* >  The complex orthogonal matrix Q is formed from V and T. */
/* > */
/* >  If TRANS='N' and SIDE='L', C is on exit replaced with Q * C. */
/* > */
/* >  If TRANS='C' and SIDE='L', C is on exit replaced with Q**H * C. */
/* > */
/* >  If TRANS='N' and SIDE='R', C is on exit replaced with C * Q. */
/* > */
/* >  If TRANS='C' and SIDE='R', C is on exit replaced with C * Q**H. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  ctpmqrt_(char *side, char *trans, integer *m, integer *n,
	 integer *k, integer *l, integer *nb, complex *v, integer *ldv, 
	complex *t, integer *ldt, complex *a, integer *lda, complex *b, 
	integer *ldb, complex *work, integer *info)
{
    /* System generated locals */
    integer v_dim1, v_offset, a_dim1, a_offset, b_dim1, b_offset, t_dim1, 
	    t_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    integer i__, ib, lb, mb, kf, ldaq;
    logical left, tran;
    integer ldvq;
    extern logical lsame_(char *, char *);
    logical right;
    extern void  xerbla_(char *, integer *), ctprfb_(
	    char *, char *, char *, char *, integer *, integer *, integer *, 
	    integer *, complex *, integer *, complex *, integer *, complex *, 
	    integer *, complex *, integer *, complex *, integer *);
    logical notran;


/*  -- LAPACK computational routine (version 3.8.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2017 */


/*  ===================================================================== */



    /* Parameter adjustments */
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --work;

    /* Function Body */
    *info = 0;
    left = lsame_(side, "L");
    right = lsame_(side, "R");
    tran = lsame_(trans, "C");
    notran = lsame_(trans, "N");

    if (left) {
	ldvq = f2cmax(1,*m);
	ldaq = f2cmax(1,*k);
    } else if (right) {
	ldvq = f2cmax(1,*n);
	ldaq = f2cmax(1,*m);
    }
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
    } else if (*l < 0 || *l > *k) {
	*info = -6;
    } else if (*nb < 1 || *nb > *k && *k > 0) {
	*info = -7;
    } else if (*ldv < ldvq) {
	*info = -9;
    } else if (*ldt < *nb) {
	*info = -11;
    } else if (*lda < ldaq) {
	*info = -13;
    } else if (*ldb < f2cmax(1,*m)) {
	*info = -15;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CTPMQRT", &i__1);
	return;
    }


    if (*m == 0 || *n == 0 || *k == 0) {
	return;
    }

    if (left && tran) {

	i__1 = *k;
	i__2 = *nb;
	for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
	    i__3 = *nb, i__4 = *k - i__ + 1;
	    ib = f2cmin(i__3,i__4);
/* Computing MIN */
	    i__3 = *m - *l + i__ + ib - 1;
	    mb = f2cmin(i__3,*m);
	    if (i__ >= *l) {
		lb = 0;
	    } else {
		lb = mb - *m + *l - i__ + 1;
	    }
	    ctprfb_("L", "C", "F", "C", &mb, n, &ib, &lb, &v[i__ * v_dim1 + 1]
		    , ldv, &t[i__ * t_dim1 + 1], ldt, &a[i__ + a_dim1], lda, &
		    b[b_offset], ldb, &work[1], &ib);
	}

    } else if (right && notran) {

	i__2 = *k;
	i__1 = *nb;
	for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1) {
/* Computing MIN */
	    i__3 = *nb, i__4 = *k - i__ + 1;
	    ib = f2cmin(i__3,i__4);
/* Computing MIN */
	    i__3 = *n - *l + i__ + ib - 1;
	    mb = f2cmin(i__3,*n);
	    if (i__ >= *l) {
		lb = 0;
	    } else {
		lb = mb - *n + *l - i__ + 1;
	    }
	    ctprfb_("R", "N", "F", "C", m, &mb, &ib, &lb, &v[i__ * v_dim1 + 1]
		    , ldv, &t[i__ * t_dim1 + 1], ldt, &a[i__ * a_dim1 + 1], 
		    lda, &b[b_offset], ldb, &work[1], m);
	}

    } else if (left && notran) {

	kf = (*k - 1) / *nb * *nb + 1;
	i__1 = -(*nb);
	for (i__ = kf; i__1 < 0 ? i__ >= 1 : i__ <= 1; i__ += i__1) {
/* Computing MIN */
	    i__2 = *nb, i__3 = *k - i__ + 1;
	    ib = f2cmin(i__2,i__3);
/* Computing MIN */
	    i__2 = *m - *l + i__ + ib - 1;
	    mb = f2cmin(i__2,*m);
	    if (i__ >= *l) {
		lb = 0;
	    } else {
		lb = mb - *m + *l - i__ + 1;
	    }
	    ctprfb_("L", "N", "F", "C", &mb, n, &ib, &lb, &v[i__ * v_dim1 + 1]
		    , ldv, &t[i__ * t_dim1 + 1], ldt, &a[i__ + a_dim1], lda, &
		    b[b_offset], ldb, &work[1], &ib);
	}

    } else if (right && tran) {

	kf = (*k - 1) / *nb * *nb + 1;
	i__1 = -(*nb);
	for (i__ = kf; i__1 < 0 ? i__ >= 1 : i__ <= 1; i__ += i__1) {
/* Computing MIN */
	    i__2 = *nb, i__3 = *k - i__ + 1;
	    ib = f2cmin(i__2,i__3);
/* Computing MIN */
	    i__2 = *n - *l + i__ + ib - 1;
	    mb = f2cmin(i__2,*n);
	    if (i__ >= *l) {
		lb = 0;
	    } else {
		lb = mb - *n + *l - i__ + 1;
	    }
	    ctprfb_("R", "C", "F", "C", m, &mb, &ib, &lb, &v[i__ * v_dim1 + 1]
		    , ldv, &t[i__ * t_dim1 + 1], ldt, &a[i__ * a_dim1 + 1], 
		    lda, &b[b_offset], ldb, &work[1], m);
	}

    }

    return;

/*     End of CTPMQRT */

} /* ctpmqrt_ */
