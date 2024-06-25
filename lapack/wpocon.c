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

static integer c__1 = 1;

/* > \brief \b ZPOCON */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZPOCON + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zpocon.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zpocon.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zpocon.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZPOCON( UPLO, N, A, LDA, ANORM, RCOND, WORK, RWORK, */
/*                          INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, LDA, N */
/*       DOUBLE PRECISION   ANORM, RCOND */
/*       DOUBLE PRECISION   RWORK( * ) */
/*       COMPLEX*16         A( LDA, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZPOCON estimates the reciprocal of the condition number (in the */
/* > 1-norm) of a complex Hermitian positive definite matrix using the */
/* > Cholesky factorization A = U**H*U or A = L*L**H computed by ZPOTRF. */
/* > */
/* > An estimate is obtained for norm(inv(A)), and the reciprocal of the */
/* > condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))). */
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
/* > \param[in] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          The triangular factor U or L from the Cholesky factorization */
/* >          A = U**H*U or A = L*L**H, as computed by ZPOTRF. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] ANORM */
/* > \verbatim */
/* >          ANORM is DOUBLE PRECISION */
/* >          The 1-norm (or infinity-norm) of the Hermitian matrix A. */
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
/* >          WORK is COMPLEX*16 array, dimension (2*N) */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is DOUBLE PRECISION array, dimension (N) */
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

/* > \ingroup complex16POcomputational */

/*  ===================================================================== */
void  wpocon_(char *uplo, integer *n, quadcomplex *a, 
	integer *lda, quadreal *anorm, quadreal *rcond, quadcomplex *
	work, quadreal *rwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1;
    quadreal d__1, d__2;

    /* Local variables */
    integer ix, kase;
    quadreal scale;
    extern logical lsame_(char *, char *);
    integer isave[3];
    logical upper;
    extern void  wlacn2_(integer *, quadcomplex *, 
	    quadcomplex *, quadreal *, integer *, integer *);
    extern quadreal qlamch_(char *);
    quadreal scalel, scaleu;
    extern void  xerbla_(char *, integer *);
    quadreal ainvnm;
    extern integer iwamax_(integer *, quadcomplex *, integer *);
    extern void  wqrscl_(integer *, quadreal *, 
	    quadcomplex *, integer *);
    char normin[1];
    quadreal smlnum;
    extern void  wlatrs_(char *, char *, char *, char *, 
	    integer *, quadcomplex *, integer *, quadcomplex *, 
	    quadreal *, quadreal *, integer *);


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
    --work;
    --rwork;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -4;
    } else if (*anorm < 0.) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZPOCON", &i__1);
	return;
    }

/*     Quick return if possible */

    *rcond = 0.;
    if (*n == 0) {
	*rcond = 1.;
	return;
    } else if (*anorm == 0.) {
	return;
    }

    smlnum = qlamch_("Safe minimum");

/*     Estimate the 1-norm of inv(A). */

    kase = 0;
    *(unsigned char *)normin = 'N';
L10:
    wlacn2_(n, &work[*n + 1], &work[1], &ainvnm, &kase, isave);
    if (kase != 0) {
	if (upper) {

/*           Multiply by inv(U**H). */

	    wlatrs_("Upper", "Conjugate transpose", "Non-unit", normin, n, &a[
		    a_offset], lda, &work[1], &scalel, &rwork[1], info);
	    *(unsigned char *)normin = 'Y';

/*           Multiply by inv(U). */

	    wlatrs_("Upper", "No transpose", "Non-unit", normin, n, &a[
		    a_offset], lda, &work[1], &scaleu, &rwork[1], info);
	} else {

/*           Multiply by inv(L). */

	    wlatrs_("Lower", "No transpose", "Non-unit", normin, n, &a[
		    a_offset], lda, &work[1], &scalel, &rwork[1], info);
	    *(unsigned char *)normin = 'Y';

/*           Multiply by inv(L**H). */

	    wlatrs_("Lower", "Conjugate transpose", "Non-unit", normin, n, &a[
		    a_offset], lda, &work[1], &scaleu, &rwork[1], info);
	}

/*        Multiply by 1/SCALE if doing so will not cause overflow. */

	scale = scalel * scaleu;
	if (scale != 1.) {
	    ix = iwamax_(n, &work[1], &c__1);
	    i__1 = ix;
	    if (scale < ((d__1 = work[i__1].r, abs(d__1)) + (d__2 = d_imag(&
		    work[ix]), abs(d__2))) * smlnum || scale == 0.) {
		goto L20;
	    }
	    wqrscl_(n, &scale, &work[1], &c__1);
	}
	goto L10;
    }

/*     Compute the estimate of the reciprocal condition number. */

    if (ainvnm != 0.) {
	*rcond = 1. / ainvnm / *anorm;
    }

L20:
    return;

/*     End of ZPOCON */

} /* wpocon_ */

