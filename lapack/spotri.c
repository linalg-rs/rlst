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

/* > \brief \b SPOTRI */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download SPOTRI + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/spotri.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/spotri.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/spotri.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE SPOTRI( UPLO, N, A, LDA, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, LDA, N */
/*       REAL               A( LDA, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > SPOTRI computes the inverse of a real symmetric positive definite */
/* > matrix A using the Cholesky factorization A = U**T*U or A = L*L**T */
/* > computed by SPOTRF. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          = 'U':  Upper triangle of A is stored; */
/* >          = 'L':  Lower triangle of A is stored. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is REAL array, dimension (LDA,N) */
/* >          On entry, the triangular factor U or L from the Cholesky */
/* >          factorization A = U**T*U or A = L*L**T, as computed by */
/* >          SPOTRF. */
/* >          On exit, the upper or lower triangle of the (symmetric) */
/* >          inverse of A, overwriting the input factor U or L. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  if INFO = i, the (i,i) element of the factor U or L is */
/* >                zero, and the inverse could not be computed. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup realPOcomputational */

/*  ===================================================================== */
void  spotri_(char *uplo, integer *n, real *a, integer *lda, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1;

    /* Local variables */
    extern logical lsame_(char *, char *);
    extern void  xerbla_(char *, integer *), slauum_(
	    char *, integer *, real *, integer *, integer *), strtri_(
	    char *, char *, integer *, real *, integer *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SPOTRI", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

/*     Invert the triangular Cholesky factor U or L. */

    strtri_(uplo, "Non-unit", n, &a[a_offset], lda, info);
    if (*info > 0) {
	return;
    }

/*     Form inv(U) * inv(U)**T or inv(L)**T * inv(L). */

    slauum_(uplo, n, &a[a_offset], lda, info);

    return;

/*     End of SPOTRI */

} /* spotri_ */

