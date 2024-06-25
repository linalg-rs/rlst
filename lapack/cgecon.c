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

/* > \brief \b CGECON */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CGECON + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/cgecon.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/cgecon.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/cgecon.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CGECON( NORM, N, A, LDA, ANORM, RCOND, WORK, RWORK, */
/*                          INFO ) */

/*       CHARACTER          NORM */
/*       INTEGER            INFO, LDA, N */
/*       REAL               ANORM, RCOND */
/*       REAL               RWORK( * ) */
/*       COMPLEX            A( LDA, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CGECON estimates the reciprocal of the condition number of a general */
/* > complex matrix A, in either the 1-norm or the infinity-norm, using */
/* > the LU factorization computed by CGETRF. */
/* > */
/* > An estimate is obtained for norm(inv(A)), and the reciprocal of the */
/* > condition number is computed as */
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
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] A */
/* > \verbatim */
/* >          A is COMPLEX array, dimension (LDA,N) */
/* >          The factors L and U from the factorization A = P*L*U */
/* >          as computed by CGETRF. */
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
/* >          ANORM is REAL */
/* >          If NORM = '1' or 'O', the 1-norm of the original matrix A. */
/* >          If NORM = 'I', the infinity-norm of the original matrix A. */
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
/* >          WORK is COMPLEX array, dimension (2*N) */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is REAL array, dimension (2*N) */
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

/* > \ingroup complexGEcomputational */

/*  ===================================================================== */
void  cgecon_(char *norm, integer *n, complex *a, integer *lda,
	 real *anorm, real *rcond, complex *work, real *rwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1;
    real r__1, r__2;

    /* Local variables */
    real sl;
    integer ix;
    real su;
    integer kase, kase1;
    real scale;
    extern logical lsame_(char *, char *);
    integer isave[3];
    extern void  clacn2_(integer *, complex *, complex *, real 
	    *, integer *, integer *);
    extern integer icamax_(integer *, complex *, integer *);
    extern real slamch_(char *);
    extern void  xerbla_(char *, integer *);
    real ainvnm;
    extern void  clatrs_(char *, char *, char *, char *, 
	    integer *, complex *, integer *, complex *, real *, real *, 
	    integer *), csrscl_(integer *, 
	    real *, complex *, integer *);
    logical onenrm;
    char normin[1];
    real smlnum;


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
    onenrm = *(unsigned char *)norm == '1' || lsame_(norm, "O");
    if (! onenrm && ! lsame_(norm, "I")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -4;
    } else if (*anorm < 0.f) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CGECON", &i__1);
	return;
    }

/*     Quick return if possible */

    *rcond = 0.f;
    if (*n == 0) {
	*rcond = 1.f;
	return;
    } else if (*anorm == 0.f) {
	return;
    }

    smlnum = slamch_("Safe minimum");

/*     Estimate the norm of inv(A). */

    ainvnm = 0.f;
    *(unsigned char *)normin = 'N';
    if (onenrm) {
	kase1 = 1;
    } else {
	kase1 = 2;
    }
    kase = 0;
L10:
    clacn2_(n, &work[*n + 1], &work[1], &ainvnm, &kase, isave);
    if (kase != 0) {
	if (kase == kase1) {

/*           Multiply by inv(L). */

	    clatrs_("Lower", "No transpose", "Unit", normin, n, &a[a_offset], 
		    lda, &work[1], &sl, &rwork[1], info);

/*           Multiply by inv(U). */

	    clatrs_("Upper", "No transpose", "Non-unit", normin, n, &a[
		    a_offset], lda, &work[1], &su, &rwork[*n + 1], info);
	} else {

/*           Multiply by inv(U**H). */

	    clatrs_("Upper", "Conjugate transpose", "Non-unit", normin, n, &a[
		    a_offset], lda, &work[1], &su, &rwork[*n + 1], info);

/*           Multiply by inv(L**H). */

	    clatrs_("Lower", "Conjugate transpose", "Unit", normin, n, &a[
		    a_offset], lda, &work[1], &sl, &rwork[1], info);
	}

/*        Divide X by 1/(SL*SU) if doing so will not cause overflow. */

	scale = sl * su;
	*(unsigned char *)normin = 'Y';
	if (scale != 1.f) {
	    ix = icamax_(n, &work[1], &c__1);
	    i__1 = ix;
	    if (scale < ((r__1 = work[i__1].r, abs(r__1)) + (r__2 = r_imag(&
		    work[ix]), abs(r__2))) * smlnum || scale == 0.f) {
		goto L20;
	    }
	    csrscl_(n, &scale, &work[1], &c__1);
	}
	goto L10;
    }

/*     Compute the estimate of the reciprocal condition number. */

    if (ainvnm != 0.f) {
	*rcond = 1.f / ainvnm / *anorm;
    }

L20:
    return;

/*     End of CGECON */

} /* cgecon_ */

