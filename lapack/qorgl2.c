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

/* > \brief \b DORGL2 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DORGL2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dorgl2.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dorgl2.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dorgl2.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DORGL2( M, N, K, A, LDA, TAU, WORK, INFO ) */

/*       INTEGER            INFO, K, LDA, M, N */
/*       DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DORGL2 generates an m by n doublereal matrix Q with orthonormal rows, */
/* > which is defined as the first m rows of a product of k elementary */
/* > reflectors of order n */
/* > */
/* >       Q  =  H(k) . . . H(2) H(1) */
/* > */
/* > as returned by DGELQF. */
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
/* >          The number of columns of the matrix Q. N >= M. */
/* > \endverbatim */
/* > */
/* > \param[in] K */
/* > \verbatim */
/* >          K is INTEGER */
/* >          The number of elementary reflectors whose product defines the */
/* >          matrix Q. M >= K >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is DOUBLE PRECISION array, dimension (LDA,N) */
/* >          On entry, the i-th row must contain the vector which defines */
/* >          the elementary reflector H(i), for i = 1,2,...,k, as returned */
/* >          by DGELQF in the first k rows of its array argument A. */
/* >          On exit, the m-by-n matrix Q. */
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
/* >          reflector H(i), as returned by DGELQF. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension (M) */
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
void  qorgl2_(integer *m, integer *n, integer *k, quadreal *
	a, integer *lda, quadreal *tau, quadreal *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    quadreal d__1;

    /* Local variables */
    integer i__, j, l;
    extern void  qscal_(integer *, quadreal *, quadreal *, 
	    integer *), qlarf_(char *, integer *, integer *, quadreal *, 
	    integer *, quadreal *, quadreal *, integer *, quadreal *), xerbla_(char *, integer *);


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
    } else if (*n < *m) {
	*info = -2;
    } else if (*k < 0 || *k > *m) {
	*info = -3;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORGL2", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*m <= 0) {
	return;
    }

    if (*k < *m) {

/*        Initialise rows k+1:m to rows of the unit matrix */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (l = *k + 1; l <= i__2; ++l) {
		a[l + j * a_dim1] = 0.;
/* L10: */
	    }
	    if (j > *k && j <= *m) {
		a[j + j * a_dim1] = 1.;
	    }
/* L20: */
	}
    }

    for (i__ = *k; i__ >= 1; --i__) {

/*        Apply H(i) to A(i:m,i:n) from the right */

	if (i__ < *n) {
	    if (i__ < *m) {
		a[i__ + i__ * a_dim1] = 1.;
		i__1 = *m - i__;
		i__2 = *n - i__ + 1;
		qlarf_("Right", &i__1, &i__2, &a[i__ + i__ * a_dim1], lda, &
			tau[i__], &a[i__ + 1 + i__ * a_dim1], lda, &work[1]);
	    }
	    i__1 = *n - i__;
	    d__1 = -tau[i__];
	    qscal_(&i__1, &d__1, &a[i__ + (i__ + 1) * a_dim1], lda);
	}
	a[i__ + i__ * a_dim1] = 1. - tau[i__];

/*        Set A(i,1:i-1) to zero */

	i__1 = i__ - 1;
	for (l = 1; l <= i__1; ++l) {
	    a[i__ + l * a_dim1] = 0.;
/* L30: */
	}
/* L40: */
    }
    return;

/*     End of DORGL2 */

} /* qorgl2_ */

