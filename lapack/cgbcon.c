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

/* > \brief \b CGBCON */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CGBCON + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/cgbcon.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/cgbcon.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/cgbcon.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CGBCON( NORM, N, KL, KU, AB, LDAB, IPIV, ANORM, RCOND, */
/*                          WORK, RWORK, INFO ) */

/*       CHARACTER          NORM */
/*       INTEGER            INFO, KL, KU, LDAB, N */
/*       REAL               ANORM, RCOND */
/*       INTEGER            IPIV( * ) */
/*       REAL               RWORK( * ) */
/*       COMPLEX            AB( LDAB, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CGBCON estimates the reciprocal of the condition number of a complex */
/* > general band matrix A, in either the 1-norm or the infinity-norm, */
/* > using the LU factorization computed by CGBTRF. */
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
/* > \param[in] KL */
/* > \verbatim */
/* >          KL is INTEGER */
/* >          The number of subdiagonals within the band of A.  KL >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] KU */
/* > \verbatim */
/* >          KU is INTEGER */
/* >          The number of superdiagonals within the band of A.  KU >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] AB */
/* > \verbatim */
/* >          AB is COMPLEX array, dimension (LDAB,N) */
/* >          Details of the LU factorization of the band matrix A, as */
/* >          computed by CGBTRF.  U is stored as an upper triangular band */
/* >          matrix with KL+KU superdiagonals in rows 1 to KL+KU+1, and */
/* >          the multipliers used during the factorization are stored in */
/* >          rows KL+KU+2 to 2*KL+KU+1. */
/* > \endverbatim */
/* > */
/* > \param[in] LDAB */
/* > \verbatim */
/* >          LDAB is INTEGER */
/* >          The leading dimension of the array AB.  LDAB >= 2*KL+KU+1. */
/* > \endverbatim */
/* > */
/* > \param[in] IPIV */
/* > \verbatim */
/* >          IPIV is INTEGER array, dimension (N) */
/* >          The pivot indices; for 1 <= i <= N, row i of the matrix was */
/* >          interchanged with row IPIV(i). */
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
/* >          RWORK is REAL array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0: if INFO = -i, the i-th argument had an illegal value */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complexGBcomputational */

/*  ===================================================================== */
void  cgbcon_(char *norm, integer *n, integer *kl, integer *ku,
	 complex *ab, integer *ldab, integer *ipiv, real *anorm, real *rcond, 
	complex *work, real *rwork, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, i__1, i__2, i__3;
    real r__1, r__2;
    complex q__1, q__2;

    /* Local variables */
    integer j;
    complex t;
    integer kd, lm, jp, ix, kase, kase1;
    real scale;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    extern logical lsame_(char *, char *);
    integer isave[3];
    extern void  caxpy_(integer *, complex *, complex *, 
	    integer *, complex *, integer *);
    logical lnoti;
    extern void  clacn2_(integer *, complex *, complex *, real 
	    *, integer *, integer *);
    extern integer icamax_(integer *, complex *, integer *);
    extern real slamch_(char *);
    extern void  clatbs_(char *, char *, char *, char *, 
	    integer *, integer *, complex *, integer *, complex *, real *, 
	    real *, integer *), xerbla_(char *
	    , integer *);
    real ainvnm;
    extern void  csrscl_(integer *, real *, complex *, integer 
	    *);
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
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    --ipiv;
    --work;
    --rwork;

    /* Function Body */
    *info = 0;
    onenrm = *(unsigned char *)norm == '1' || lsame_(norm, "O");
    if (! onenrm && ! lsame_(norm, "I")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*kl < 0) {
	*info = -3;
    } else if (*ku < 0) {
	*info = -4;
    } else if (*ldab < (*kl << 1) + *ku + 1) {
	*info = -6;
    } else if (*anorm < 0.f) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CGBCON", &i__1);
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
    kd = *kl + *ku + 1;
    lnoti = *kl > 0;
    kase = 0;
L10:
    clacn2_(n, &work[*n + 1], &work[1], &ainvnm, &kase, isave);
    if (kase != 0) {
	if (kase == kase1) {

/*           Multiply by inv(L). */

	    if (lnoti) {
		i__1 = *n - 1;
		for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
		    i__2 = *kl, i__3 = *n - j;
		    lm = f2cmin(i__2,i__3);
		    jp = ipiv[j];
		    i__2 = jp;
		    t.r = work[i__2].r, t.i = work[i__2].i;
		    if (jp != j) {
			i__2 = jp;
			i__3 = j;
			work[i__2].r = work[i__3].r, work[i__2].i = work[i__3]
				.i;
			i__2 = j;
			work[i__2].r = t.r, work[i__2].i = t.i;
		    }
		    q__1.r = -t.r, q__1.i = -t.i;
		    caxpy_(&lm, &q__1, &ab[kd + 1 + j * ab_dim1], &c__1, &
			    work[j + 1], &c__1);
/* L20: */
		}
	    }

/*           Multiply by inv(U). */

	    i__1 = *kl + *ku;
	    clatbs_("Upper", "No transpose", "Non-unit", normin, n, &i__1, &
		    ab[ab_offset], ldab, &work[1], &scale, &rwork[1], info);
	} else {

/*           Multiply by inv(U**H). */

	    i__1 = *kl + *ku;
	    clatbs_("Upper", "Conjugate transpose", "Non-unit", normin, n, &
		    i__1, &ab[ab_offset], ldab, &work[1], &scale, &rwork[1], 
		    info);

/*           Multiply by inv(L**H). */

	    if (lnoti) {
		for (j = *n - 1; j >= 1; --j) {
/* Computing MIN */
		    i__1 = *kl, i__2 = *n - j;
		    lm = f2cmin(i__1,i__2);
		    i__1 = j;
		    i__2 = j;
		    cdotc_(&q__2, &lm, &ab[kd + 1 + j * ab_dim1], &c__1, &
			    work[j + 1], &c__1);
		    q__1.r = work[i__2].r - q__2.r, q__1.i = work[i__2].i - 
			    q__2.i;
		    work[i__1].r = q__1.r, work[i__1].i = q__1.i;
		    jp = ipiv[j];
		    if (jp != j) {
			i__1 = jp;
			t.r = work[i__1].r, t.i = work[i__1].i;
			i__1 = jp;
			i__2 = j;
			work[i__1].r = work[i__2].r, work[i__1].i = work[i__2]
				.i;
			i__1 = j;
			work[i__1].r = t.r, work[i__1].i = t.i;
		    }
/* L30: */
		}
	    }
	}

/*        Divide X by 1/SCALE if doing so will not cause overflow. */

	*(unsigned char *)normin = 'Y';
	if (scale != 1.f) {
	    ix = icamax_(n, &work[1], &c__1);
	    i__1 = ix;
	    if (scale < ((r__1 = work[i__1].r, abs(r__1)) + (r__2 = r_imag(&
		    work[ix]), abs(r__2))) * smlnum || scale == 0.f) {
		goto L40;
	    }
	    csrscl_(n, &scale, &work[1], &c__1);
	}
	goto L10;
    }

/*     Compute the estimate of the reciprocal condition number. */

    if (ainvnm != 0.f) {
	*rcond = 1.f / ainvnm / *anorm;
    }

L40:
    return;

/*     End of CGBCON */

} /* cgbcon_ */

