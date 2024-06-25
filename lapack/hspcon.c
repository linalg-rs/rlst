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

/* Table of constant values */

static integer c__1 = 1;

/* > \brief \b DSPCON */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DSPCON + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dspcon.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dspcon.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dspcon.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DSPCON( UPLO, N, AP, IPIV, ANORM, RCOND, WORK, IWORK, */
/*                          INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, N */
/*       DOUBLE PRECISION   ANORM, RCOND */
/*       INTEGER            IPIV( * ), IWORK( * ) */
/*       DOUBLE PRECISION   AP( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DSPCON estimates the reciprocal of the condition number (in the */
/* > 1-norm) of a doublereal symmetric packed matrix A using the factorization */
/* > A = U*D*U**T or A = L*D*L**T computed by DSPTRF. */
/* > */
/* > An estimate is obtained for norm(inv(A)), and the reciprocal of the */
/* > condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))). */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          Specifies whether the details of the factorization are stored */
/* >          as an upper or lower triangular matrix. */
/* >          = 'U':  Upper triangular, form is A = U*D*U**T; */
/* >          = 'L':  Lower triangular, form is A = L*D*L**T. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] AP */
/* > \verbatim */
/* >          AP is DOUBLE PRECISION array, dimension (N*(N+1)/2) */
/* >          The block diagonal matrix D and the multipliers used to */
/* >          obtain the factor U or L as computed by DSPTRF, stored as a */
/* >          packed triangular matrix. */
/* > \endverbatim */
/* > */
/* > \param[in] IPIV */
/* > \verbatim */
/* >          IPIV is INTEGER array, dimension (N) */
/* >          Details of the interchanges and the block structure of D */
/* >          as determined by DSPTRF. */
/* > \endverbatim */
/* > */
/* > \param[in] ANORM */
/* > \verbatim */
/* >          ANORM is DOUBLE PRECISION */
/* >          The 1-norm of the original matrix A. */
/* > \endverbatim */
/* > */
/* > \param[out] RCOND */
/* > \verbatim */
/* >          RCOND is DOUBLE PRECISION */
/* >          The reciprocal of the condition number of the matrix A, */
/* >          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is an */
/* >          estimate of the 1-norm of inv(A) computed in this routine. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension (2*N) */
/* > \endverbatim */
/* > */
/* > \param[out] IWORK */
/* > \verbatim */
/* >          IWORK is INTEGER array, dimension (N) */
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

/* > \ingroup doubleOTHERcomputational */

/*  ===================================================================== */
void  hspcon_(char *uplo, integer *n, halfreal *ap, integer *
	ipiv, halfreal *anorm, halfreal *rcond, halfreal *work, integer 
	*iwork, integer *info)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__, ip, kase;
    extern logical lsame_(char *, char *);
    integer isave[3];
    logical upper;
    extern void  hlacn2_(integer *, halfreal *, halfreal *,
	     integer *, halfreal *, integer *, integer *), xerbla_(char *, 
	    integer *);
    halfreal ainvnm;
    extern void  hsptrs_(char *, integer *, integer *, 
	    halfreal *, integer *, halfreal *, integer *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    --iwork;
    --work;
    --ipiv;
    --ap;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*anorm < 0.) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSPCON", &i__1);
	return;
    }

/*     Quick return if possible */

    *rcond = 0.;
    if (*n == 0) {
	*rcond = 1.;
	return;
    } else if (*anorm <= 0.) {
	return;
    }

/*     Check that the diagonal matrix D is nonsingular. */

    if (upper) {

/*        Upper triangular storage: examine D from bottom to top */

	ip = *n * (*n + 1) / 2;
	for (i__ = *n; i__ >= 1; --i__) {
	    if (ipiv[i__] > 0 && ap[ip] == 0.) {
		return;
	    }
	    ip -= i__;
/* L10: */
	}
    } else {

/*        Lower triangular storage: examine D from top to bottom. */

	ip = 1;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (ipiv[i__] > 0 && ap[ip] == 0.) {
		return;
	    }
	    ip = ip + *n - i__ + 1;
/* L20: */
	}
    }

/*     Estimate the 1-norm of the inverse. */

    kase = 0;
L30:
    hlacn2_(n, &work[*n + 1], &work[1], &iwork[1], &ainvnm, &kase, isave);
    if (kase != 0) {

/*        Multiply by inv(L*D*L**T) or inv(U*D*U**T). */

	hsptrs_(uplo, n, &c__1, &ap[1], &ipiv[1], &work[1], n, info);
	goto L30;
    }

/*     Compute the estimate of the reciprocal condition number. */

    if (ainvnm != 0.) {
	*rcond = 1. / ainvnm / *anorm;
    }

    return;

/*     End of DSPCON */

} /* hspcon_ */

