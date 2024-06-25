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

#define __LAPACK_PRECISION_HALF
#include "f2c.h"

/* > \brief \b ZGEQRT */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZGEQRT + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zgeqrt.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zgeqrt.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zgeqrt.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZGEQRT( M, N, NB, A, LDA, T, LDT, WORK, INFO ) */

/*       INTEGER INFO, LDA, LDT, M, N, NB */
/*       COMPLEX*16 A( LDA, * ), T( LDT, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZGEQRT computes a blocked QR factorization of a complex M-by-N matrix A */
/* > using the compact WY representation of Q. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix A.  M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] NB */
/* > \verbatim */
/* >          NB is INTEGER */
/* >          The block size to be used in the blocked QR.  MIN(M,N) >= NB >= 1. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          On entry, the M-by-N matrix A. */
/* >          On exit, the elements on and above the diagonal of the array */
/* >          contain the f2cmin(M,N)-by-N upper trapezoidal matrix R (R is */
/* >          upper triangular if M >= N); the elements below the diagonal */
/* >          are the columns of V. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[out] T */
/* > \verbatim */
/* >          T is COMPLEX*16 array, dimension (LDT,MIN(M,N)) */
/* >          The upper triangular block reflectors stored in compact form */
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
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX*16 array, dimension (NB*N) */
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

/* > \date June 2017 */

/* > \ingroup complex16GEcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  The matrix V stores the elementary reflectors H(i) in the i-th column */
/* >  below the diagonal. For example, if M=5 and N=3, the matrix V is */
/* > */
/* >               V = (  1       ) */
/* >                   ( v1  1    ) */
/* >                   ( v1 v2  1 ) */
/* >                   ( v1 v2 v3 ) */
/* >                   ( v1 v2 v3 ) */
/* > */
/* >  where the vi's represent the vectors which define H(i), which are returned */
/* >  in the matrix A.  The 1's along the diagonal of V are not stored in A. */
/* > */
/* >  Let K=MIN(M,N).  The number of blocks is B = ceiling(K/NB), where each */
/* >  block is of order NB except for the last block, which is of order */
/* >  IB = K - (B-1)*NB.  For each of the B blocks, a upper triangular block */
/* >  reflector factor is computed: T1, T2, ..., TB.  The NB-by-NB (and IB-by-IB */
/* >  for the last block) T's are stored in the NB-by-K matrix T as */
/* > */
/* >               T = (T1 T2 ... TB). */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  kgeqrt_(integer *m, integer *n, integer *nb, 
	halfcomplex *a, integer *lda, halfcomplex *t, integer *ldt, 
	halfcomplex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, t_dim1, t_offset, i__1, i__2, i__3, i__4, i__5;

    /* Local variables */
    integer i__, k, ib, iinfo;
    extern void  xerbla_(char *, integer *), klarfb_(
	    char *, char *, char *, char *, integer *, integer *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, integer *), kgeqrt2_(integer *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, integer *, integer *)
	    , kgeqrt3_(integer *, integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *, integer *);


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
    --work;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nb < 1 || *nb > f2cmin(*m,*n) && f2cmin(*m,*n) > 0) {
	*info = -3;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -5;
    } else if (*ldt < *nb) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGEQRT", &i__1);
	return;
    }

/*     Quick return if possible */

    k = f2cmin(*m,*n);
    if (k == 0) {
	return;
    }

/*     Blocked loop of length K */

    i__1 = k;
    i__2 = *nb;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
	i__3 = k - i__ + 1;
	ib = f2cmin(i__3,*nb);

/*     Compute the QR factorization of the current block A(I:M,I:I+IB-1) */

	if (TRUE_) {
	    i__3 = *m - i__ + 1;
	    kgeqrt3_(&i__3, &ib, &a[i__ + i__ * a_dim1], lda, &t[i__ * t_dim1 
		    + 1], ldt, &iinfo);
	} else {
	    i__3 = *m - i__ + 1;
	    kgeqrt2_(&i__3, &ib, &a[i__ + i__ * a_dim1], lda, &t[i__ * t_dim1 
		    + 1], ldt, &iinfo);
	}
	if (i__ + ib <= *n) {

/*     Update by applying H**H to A(I:M,I+IB:N) from the left */

	    i__3 = *m - i__ + 1;
	    i__4 = *n - i__ - ib + 1;
	    i__5 = *n - i__ - ib + 1;
	    klarfb_("L", "C", "F", "C", &i__3, &i__4, &ib, &a[i__ + i__ * 
		    a_dim1], lda, &t[i__ * t_dim1 + 1], ldt, &a[i__ + (i__ + 
		    ib) * a_dim1], lda, &work[1], &i__5);
	}
    }
    return;

/*     End of ZGEQRT */

} /* kgeqrt_ */

