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

#define __LAPACK_PRECISION_DOUBLE
#include "f2c.h"

/* > \brief \b DGEMQRT */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DGEMQRT + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgemqrt
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgemqrt
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgemqrt
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DGEMQRT( SIDE, TRANS, M, N, K, NB, V, LDV, T, LDT, */
/*                          C, LDC, WORK, INFO ) */

/*       CHARACTER SIDE, TRANS */
/*       INTEGER   INFO, K, LDV, LDC, M, N, NB, LDT */
/*       DOUBLE PRECISION V( LDV, * ), C( LDC, * ), T( LDT, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DGEMQRT overwrites the general real M-by-N matrix C with */
/* > */
/* >                 SIDE = 'L'     SIDE = 'R' */
/* > TRANS = 'N':      Q C            C Q */
/* > TRANS = 'T':   Q**T C            C Q**T */
/* > */
/* > where Q is a real orthogonal matrix defined as the product of K */
/* > elementary reflectors: */
/* > */
/* >       Q = H(1) H(2) . . . H(K) = I - V T V**T */
/* > */
/* > generated using the compact WY representation as returned by DGEQRT. */
/* > */
/* > Q is of order M if SIDE = 'L' and of order N  if SIDE = 'R'. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] SIDE */
/* > \verbatim */
/* >          SIDE is CHARACTER*1 */
/* >          = 'L': apply Q or Q**T from the Left; */
/* >          = 'R': apply Q or Q**T from the Right. */
/* > \endverbatim */
/* > */
/* > \param[in] TRANS */
/* > \verbatim */
/* >          TRANS is CHARACTER*1 */
/* >          = 'N':  No transpose, apply Q; */
/* >          = 'C':  Transpose, apply Q**T. */
/* > \endverbatim */
/* > */
/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix C. M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix C. N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] K */
/* > \verbatim */
/* >          K is INTEGER */
/* >          The number of elementary reflectors whose product defines */
/* >          the matrix Q. */
/* >          If SIDE = 'L', M >= K >= 0; */
/* >          if SIDE = 'R', N >= K >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] NB */
/* > \verbatim */
/* >          NB is INTEGER */
/* >          The block size used for the storage of T.  K >= NB >= 1. */
/* >          This must be the same value of NB used to generate T */
/* >          in CGEQRT. */
/* > \endverbatim */
/* > */
/* > \param[in] V */
/* > \verbatim */
/* >          V is DOUBLE PRECISION array, dimension (LDV,K) */
/* >          The i-th column must contain the vector which defines the */
/* >          elementary reflector H(i), for i = 1,2,...,k, as returned by */
/* >          CGEQRT in the first K columns of its array argument A. */
/* > \endverbatim */
/* > */
/* > \param[in] LDV */
/* > \verbatim */
/* >          LDV is INTEGER */
/* >          The leading dimension of the array V. */
/* >          If SIDE = 'L', LDA >= f2cmax(1,M); */
/* >          if SIDE = 'R', LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] T */
/* > \verbatim */
/* >          T is DOUBLE PRECISION array, dimension (LDT,K) */
/* >          The upper triangular factors of the block reflectors */
/* >          as returned by CGEQRT, stored as a NB-by-N matrix. */
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
/* >          C is DOUBLE PRECISION array, dimension (LDC,N) */
/* >          On entry, the M-by-N matrix C. */
/* >          On exit, C is overwritten by Q C, Q**T C, C Q**T or C Q. */
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
/* >          WORK is DOUBLE PRECISION array. The dimension of */
/* >          WORK is N*NB if SIDE = 'L', or  M*NB if SIDE = 'R'. */
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

/* > \date December 2016 */

/* > \ingroup doubleGEcomputational */

/*  ===================================================================== */
void  dgemqrt_(char *side, char *trans, integer *m, integer *n,
	 integer *k, integer *nb, doublereal *v, integer *ldv, doublereal *t, 
	integer *ldt, doublereal *c__, integer *ldc, doublereal *work, 
	integer *info)
{
    /* System generated locals */
    integer v_dim1, v_offset, c_dim1, c_offset, t_dim1, t_offset, i__1, i__2, 
	    i__3, i__4;

    /* Local variables */
    integer i__, q, ib, kf;
    logical left, tran;
    extern logical lsame_(char *, char *);
    logical right;
    extern void  dlarfb_(char *, char *, char *, char *, 
	    integer *, integer *, integer *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    integer *), xerbla_(char *, 
	    integer *);
    logical notran;
    integer ldwork;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */



    /* Parameter adjustments */
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    left = lsame_(side, "L");
    right = lsame_(side, "R");
    tran = lsame_(trans, "T");
    notran = lsame_(trans, "N");

    if (left) {
	ldwork = f2cmax(1,*n);
	q = *m;
    } else if (right) {
	ldwork = f2cmax(1,*m);
	q = *n;
    }
    if (! left && ! right) {
	*info = -1;
    } else if (! tran && ! notran) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > q) {
	*info = -5;
    } else if (*nb < 1 || *nb > *k && *k > 0) {
	*info = -6;
    } else if (*ldv < f2cmax(1,q)) {
	*info = -8;
    } else if (*ldt < *nb) {
	*info = -10;
    } else if (*ldc < f2cmax(1,*m)) {
	*info = -12;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGEMQRT", &i__1);
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
	    i__3 = *m - i__ + 1;
	    dlarfb_("L", "T", "F", "C", &i__3, n, &ib, &v[i__ + i__ * v_dim1],
		     ldv, &t[i__ * t_dim1 + 1], ldt, &c__[i__ + c_dim1], ldc, 
		    &work[1], &ldwork);
	}

    } else if (right && notran) {

	i__2 = *k;
	i__1 = *nb;
	for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1) {
/* Computing MIN */
	    i__3 = *nb, i__4 = *k - i__ + 1;
	    ib = f2cmin(i__3,i__4);
	    i__3 = *n - i__ + 1;
	    dlarfb_("R", "N", "F", "C", m, &i__3, &ib, &v[i__ + i__ * v_dim1],
		     ldv, &t[i__ * t_dim1 + 1], ldt, &c__[i__ * c_dim1 + 1], 
		    ldc, &work[1], &ldwork);
	}

    } else if (left && notran) {

	kf = (*k - 1) / *nb * *nb + 1;
	i__1 = -(*nb);
	for (i__ = kf; i__1 < 0 ? i__ >= 1 : i__ <= 1; i__ += i__1) {
/* Computing MIN */
	    i__2 = *nb, i__3 = *k - i__ + 1;
	    ib = f2cmin(i__2,i__3);
	    i__2 = *m - i__ + 1;
	    dlarfb_("L", "N", "F", "C", &i__2, n, &ib, &v[i__ + i__ * v_dim1],
		     ldv, &t[i__ * t_dim1 + 1], ldt, &c__[i__ + c_dim1], ldc, 
		    &work[1], &ldwork);
	}

    } else if (right && tran) {

	kf = (*k - 1) / *nb * *nb + 1;
	i__1 = -(*nb);
	for (i__ = kf; i__1 < 0 ? i__ >= 1 : i__ <= 1; i__ += i__1) {
/* Computing MIN */
	    i__2 = *nb, i__3 = *k - i__ + 1;
	    ib = f2cmin(i__2,i__3);
	    i__2 = *n - i__ + 1;
	    dlarfb_("R", "T", "F", "C", m, &i__2, &ib, &v[i__ + i__ * v_dim1],
		     ldv, &t[i__ * t_dim1 + 1], ldt, &c__[i__ * c_dim1 + 1], 
		    ldc, &work[1], &ldwork);
	}

    }

    return;

/*     End of DGEMQRT */

} /* dgemqrt_ */

