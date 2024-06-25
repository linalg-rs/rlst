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

/* Table of constant values */

static integer c__1 = 1;

/* > \brief \b DORG2L generates all or part of the orthogonal matrix Q from a QL factorization determined by s
geqlf (unblocked algorithm). */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DORG2L + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dorg2l.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dorg2l.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dorg2l.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DORG2L( M, N, K, A, LDA, TAU, WORK, INFO ) */

/*       INTEGER            INFO, K, LDA, M, N */
/*       DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DORG2L generates an m by n real matrix Q with orthonormal columns, */
/* > which is defined as the last n columns of a product of k elementary */
/* > reflectors of order m */
/* > */
/* >       Q  =  H(k) . . . H(2) H(1) */
/* > */
/* > as returned by DGEQLF. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix Q. M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix Q. M >= N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] K */
/* > \verbatim */
/* >          K is INTEGER */
/* >          The number of elementary reflectors whose product defines the */
/* >          matrix Q. N >= K >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is DOUBLE PRECISION array, dimension (LDA,N) */
/* >          On entry, the (n-k+i)-th column must contain the vector which */
/* >          defines the elementary reflector H(i), for i = 1,2,...,k, as */
/* >          returned by DGEQLF in the last k columns of its array */
/* >          argument A. */
/* >          On exit, the m by n matrix Q. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The first dimension of the array A. LDA >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[in] TAU */
/* > \verbatim */
/* >          TAU is DOUBLE PRECISION array, dimension (K) */
/* >          TAU(i) must contain the scalar factor of the elementary */
/* >          reflector H(i), as returned by DGEQLF. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0: successful exit */
/* >          < 0: if INFO = -i, the i-th argument has an illegal value */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup doubleOTHERcomputational */

/*  ===================================================================== */
void  dorg2l_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublereal d__1;

    /* Local variables */
    integer i__, j, l, ii;
    extern void  dscal_(integer *, doublereal *, doublereal *, 
	    integer *), dlarf_(char *, integer *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *, doublereal *), xerbla_(char *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0 || *n > *m) {
	*info = -2;
    } else if (*k < 0 || *k > *n) {
	*info = -3;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORG2L", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n <= 0) {
	return;
    }

/*     Initialise columns 1:n-k to columns of the unit matrix */

    i__1 = *n - *k;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (l = 1; l <= i__2; ++l) {
	    a[l + j * a_dim1] = 0.;
/* L10: */
	}
	a[*m - *n + j + j * a_dim1] = 1.;
/* L20: */
    }

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ii = *n - *k + i__;

/*        Apply H(i) to A(1:m-k+i,1:n-k+i) from the left */

	a[*m - *n + ii + ii * a_dim1] = 1.;
	i__2 = *m - *n + ii;
	i__3 = ii - 1;
	dlarf_("Left", &i__2, &i__3, &a[ii * a_dim1 + 1], &c__1, &tau[i__], &
		a[a_offset], lda, &work[1]);
	i__2 = *m - *n + ii - 1;
	d__1 = -tau[i__];
	dscal_(&i__2, &d__1, &a[ii * a_dim1 + 1], &c__1);
	a[*m - *n + ii + ii * a_dim1] = 1. - tau[i__];

/*        Set A(m-k+i+1:m,n-k+i) to zero */

	i__2 = *m;
	for (l = *m - *n + ii + 1; l <= i__2; ++l) {
	    a[l + ii * a_dim1] = 0.;
/* L30: */
	}
/* L40: */
    }
    return;

/*     End of DORG2L */

} /* dorg2l_ */

