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

/* Table of constant values */

static integer c__1 = 1;

/* > \brief \b STPCON */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download STPCON + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/stpcon.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/stpcon.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/stpcon.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE STPCON( NORM, UPLO, DIAG, N, AP, RCOND, WORK, IWORK, */
/*                          INFO ) */

/*       CHARACTER          DIAG, NORM, UPLO */
/*       INTEGER            INFO, N */
/*       REAL               RCOND */
/*       INTEGER            IWORK( * ) */
/*       REAL               AP( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > STPCON estimates the reciprocal of the condition number of a packed */
/* > triangular matrix A, in either the 1-norm or the infinity-norm. */
/* > */
/* > The norm of A is computed and an estimate is obtained for */
/* > norm(inv(A)), then the reciprocal of the condition number is */
/* > computed as */
/* >    RCOND = 1 / ( norm(A) * norm(inv(A)) ). */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] NORM */
/* > \verbatim */
/* >          NORM is CHARACTER*1 */
/* >          Specifies whether the 1-norm condition number or the */
/* >          infinity-norm condition number is required: */
/* >          = '1' or 'O':  1-norm; */
/* >          = 'I':         Infinity-norm. */
/* > \endverbatim */
/* > */
/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          = 'U':  A is upper triangular; */
/* >          = 'L':  A is lower triangular. */
/* > \endverbatim */
/* > */
/* > \param[in] DIAG */
/* > \verbatim */
/* >          DIAG is CHARACTER*1 */
/* >          = 'N':  A is non-unit triangular; */
/* >          = 'U':  A is unit triangular. */
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
/* >          AP is REAL array, dimension (N*(N+1)/2) */
/* >          The upper or lower triangular matrix A, packed columnwise in */
/* >          a linear array.  The j-th column of A is stored in the array */
/* >          AP as follows: */
/* >          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/* >          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n. */
/* >          If DIAG = 'U', the diagonal elements of A are not referenced */
/* >          and are assumed to be 1. */
/* > \endverbatim */
/* > */
/* > \param[out] RCOND */
/* > \verbatim */
/* >          RCOND is REAL */
/* >          The reciprocal of the condition number of the matrix A, */
/* >          computed as RCOND = 1/(norm(A) * norm(inv(A))). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is REAL array, dimension (3*N) */
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

/* > \ingroup realOTHERcomputational */

/*  ===================================================================== */
void  stpcon_(char *norm, char *uplo, char *diag, integer *n, 
	real *ap, real *rcond, real *work, integer *iwork, integer *info)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Local variables */
    integer ix, kase, kase1;
    real scale;
    extern logical lsame_(char *, char *);
    integer isave[3];
    real anorm;
    extern void  srscl_(integer *, real *, real *, integer *);
    logical upper;
    real xnorm;
    extern void  slacn2_(integer *, real *, real *, integer *, 
	    real *, integer *, integer *);
    extern real slamch_(char *);
    extern void  xerbla_(char *, integer *);
    extern integer isamax_(integer *, real *, integer *);
    real ainvnm;
    logical onenrm;
    extern real slantp_(char *, char *, char *, integer *, real *, real *);
    char normin[1];
    extern void  slatps_(char *, char *, char *, char *, 
	    integer *, real *, real *, real *, real *, integer *);
    real smlnum;
    logical nounit;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    --iwork;
    --work;
    --ap;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    onenrm = *(unsigned char *)norm == '1' || lsame_(norm, "O");
    nounit = lsame_(diag, "N");

    if (! onenrm && ! lsame_(norm, "I")) {
	*info = -1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (! nounit && ! lsame_(diag, "U")) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("STPCON", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	*rcond = 1.f;
	return;
    }

    *rcond = 0.f;
    smlnum = slamch_("Safe minimum") * (real) f2cmax(1,*n);

/*     Compute the norm of the triangular matrix A. */

    anorm = slantp_(norm, uplo, diag, n, &ap[1], &work[1]);

/*     Continue only if ANORM > 0. */

    if (anorm > 0.f) {

/*        Estimate the norm of the inverse of A. */

	ainvnm = 0.f;
	*(unsigned char *)normin = 'N';
	if (onenrm) {
	    kase1 = 1;
	} else {
	    kase1 = 2;
	}
	kase = 0;
L10:
	slacn2_(n, &work[*n + 1], &work[1], &iwork[1], &ainvnm, &kase, isave);
	if (kase != 0) {
	    if (kase == kase1) {

/*              Multiply by inv(A). */

		slatps_(uplo, "No transpose", diag, normin, n, &ap[1], &work[
			1], &scale, &work[(*n << 1) + 1], info);
	    } else {

/*              Multiply by inv(A**T). */

		slatps_(uplo, "Transpose", diag, normin, n, &ap[1], &work[1], 
			&scale, &work[(*n << 1) + 1], info);
	    }
	    *(unsigned char *)normin = 'Y';

/*           Multiply by 1/SCALE if doing so will not cause overflow. */

	    if (scale != 1.f) {
		ix = isamax_(n, &work[1], &c__1);
		xnorm = (r__1 = work[ix], abs(r__1));
		if (scale < xnorm * smlnum || scale == 0.f) {
		    goto L20;
		}
		srscl_(n, &scale, &work[1], &c__1);
	    }
	    goto L10;
	}

/*        Compute the estimate of the reciprocal condition number. */

	if (ainvnm != 0.f) {
	    *rcond = 1.f / anorm / ainvnm;
	}
    }

L20:
    return;

/*     End of STPCON */

} /* stpcon_ */

