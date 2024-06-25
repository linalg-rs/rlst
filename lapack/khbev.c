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

static halfreal c_b11 = 1.;
static integer c__1 = 1;

/* > \brief <b> ZHBEV computes the eigenvalues and, optionally, the left and/or right eigenvectors for OTHER m
atrices</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZHBEV + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zhbev.f
"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zhbev.f
"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zhbev.f
"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZHBEV( JOBZ, UPLO, N, KD, AB, LDAB, W, Z, LDZ, WORK, */
/*                         RWORK, INFO ) */

/*       CHARACTER          JOBZ, UPLO */
/*       INTEGER            INFO, KD, LDAB, LDZ, N */
/*       DOUBLE PRECISION   RWORK( * ), W( * ) */
/*       COMPLEX*16         AB( LDAB, * ), WORK( * ), Z( LDZ, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZHBEV computes all the eigenvalues and, optionally, eigenvectors of */
/* > a complex Hermitian band matrix A. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] JOBZ */
/* > \verbatim */
/* >          JOBZ is CHARACTER*1 */
/* >          = 'N':  Compute eigenvalues only; */
/* >          = 'V':  Compute eigenvalues and eigenvectors. */
/* > \endverbatim */
/* > */
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
/* > \param[in] KD */
/* > \verbatim */
/* >          KD is INTEGER */
/* >          The number of superdiagonals of the matrix A if UPLO = 'U', */
/* >          or the number of subdiagonals if UPLO = 'L'.  KD >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] AB */
/* > \verbatim */
/* >          AB is COMPLEX*16 array, dimension (LDAB, N) */
/* >          On entry, the upper or lower triangle of the Hermitian band */
/* >          matrix A, stored in the first KD+1 rows of the array.  The */
/* >          j-th column of A is stored in the j-th column of the array AB */
/* >          as follows: */
/* >          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for f2cmax(1,j-kd)<=i<=j; */
/* >          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=f2cmin(n,j+kd). */
/* > */
/* >          On exit, AB is overwritten by values generated during the */
/* >          reduction to tridiagonal form.  If UPLO = 'U', the first */
/* >          superdiagonal and the diagonal of the tridiagonal matrix T */
/* >          are returned in rows KD and KD+1 of AB, and if UPLO = 'L', */
/* >          the diagonal and first subdiagonal of T are returned in the */
/* >          first two rows of AB. */
/* > \endverbatim */
/* > */
/* > \param[in] LDAB */
/* > \verbatim */
/* >          LDAB is INTEGER */
/* >          The leading dimension of the array AB.  LDAB >= KD + 1. */
/* > \endverbatim */
/* > */
/* > \param[out] W */
/* > \verbatim */
/* >          W is DOUBLE PRECISION array, dimension (N) */
/* >          If INFO = 0, the eigenvalues in ascending order. */
/* > \endverbatim */
/* > */
/* > \param[out] Z */
/* > \verbatim */
/* >          Z is COMPLEX*16 array, dimension (LDZ, N) */
/* >          If JOBZ = 'V', then if INFO = 0, Z contains the orthonormal */
/* >          eigenvectors of the matrix A, with the i-th column of Z */
/* >          holding the eigenvector associated with W(i). */
/* >          If JOBZ = 'N', then Z is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDZ */
/* > \verbatim */
/* >          LDZ is INTEGER */
/* >          The leading dimension of the array Z.  LDZ >= 1, and if */
/* >          JOBZ = 'V', LDZ >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX*16 array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is DOUBLE PRECISION array, dimension (f2cmax(1,3*N-2)) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit. */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/* >          > 0:  if INFO = i, the algorithm failed to converge; i */
/* >                off-diagonal elements of an intermediate tridiagonal */
/* >                form did not converge to zero. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16OTHEReigen */

/*  ===================================================================== */
void  khbev_(char *jobz, char *uplo, integer *n, integer *kd, 
	halfcomplex *ab, integer *ldab, halfreal *w, halfcomplex *z__, 
	integer *ldz, halfcomplex *work, halfreal *rwork, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, z_dim1, z_offset, i__1;
    halfreal d__1;

    /* Local variables */
    halfreal eps;
    integer inde;
    halfreal anrm;
    integer imax;
    halfreal rmin, rmax;
    extern void  hscal_(integer *, halfreal *, halfreal *, 
	    integer *);
    halfreal sigma;
    extern logical lsame_(char *, char *);
    integer iinfo;
    logical lower, wantz;
    extern halfreal hlamch_(char *);
    integer iscale;
    halfreal safmin;
    extern halfreal klanhb_(char *, char *, integer *, integer *, 
	    halfcomplex *, integer *, halfreal *);
    extern void  xerbla_(char *, integer *);
    halfreal bignum;
    extern void  hsterf_(integer *, halfreal *, halfreal *,
	     integer *), klascl_(char *, integer *, integer *, halfreal *, 
	    halfreal *, integer *, integer *, halfcomplex *, integer *, 
	    integer *), khbtrd_(char *, char *, integer *, integer *, 
	    halfcomplex *, integer *, halfreal *, halfreal *, 
	    halfcomplex *, integer *, halfcomplex *, integer *);
    integer indrwk;
    halfreal smlnum;
    extern void  ksteqr_(char *, integer *, halfreal *, 
	    halfreal *, halfcomplex *, integer *, halfreal *, integer *);


/*  -- LAPACK driver routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    --w;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;
    --rwork;

    /* Function Body */
    wantz = lsame_(jobz, "V");
    lower = lsame_(uplo, "L");

    *info = 0;
    if (! (wantz || lsame_(jobz, "N"))) {
	*info = -1;
    } else if (! (lower || lsame_(uplo, "U"))) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*kd < 0) {
	*info = -4;
    } else if (*ldab < *kd + 1) {
	*info = -6;
    } else if (*ldz < 1 || wantz && *ldz < *n) {
	*info = -9;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZHBEV ", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

    if (*n == 1) {
	if (lower) {
	    i__1 = ab_dim1 + 1;
	    w[1] = ab[i__1].r;
	} else {
	    i__1 = *kd + 1 + ab_dim1;
	    w[1] = ab[i__1].r;
	}
	if (wantz) {
	    i__1 = z_dim1 + 1;
	    z__[i__1].r = 1., z__[i__1].i = 0.;
	}
	return;
    }

/*     Get machine constants. */

    safmin = hlamch_("Safe minimum");
    eps = hlamch_("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = M(sqrt)(smlnum);
    rmax = M(sqrt)(bignum);

/*     Scale matrix to allowable range, if necessary. */

    anrm = klanhb_("M", uplo, n, kd, &ab[ab_offset], ldab, &rwork[1]);
    iscale = 0;
    if (anrm > 0. && anrm < rmin) {
	iscale = 1;
	sigma = rmin / anrm;
    } else if (anrm > rmax) {
	iscale = 1;
	sigma = rmax / anrm;
    }
    if (iscale == 1) {
	if (lower) {
	    klascl_("B", kd, kd, &c_b11, &sigma, n, n, &ab[ab_offset], ldab, 
		    info);
	} else {
	    klascl_("Q", kd, kd, &c_b11, &sigma, n, n, &ab[ab_offset], ldab, 
		    info);
	}
    }

/*     Call ZHBTRD to reduce Hermitian band matrix to tridiagonal form. */

    inde = 1;
    khbtrd_(jobz, uplo, n, kd, &ab[ab_offset], ldab, &w[1], &rwork[inde], &
	    z__[z_offset], ldz, &work[1], &iinfo);

/*     For eigenvalues only, call DSTERF.  For eigenvectors, call ZSTEQR. */

    if (! wantz) {
	hsterf_(n, &w[1], &rwork[inde], info);
    } else {
	indrwk = inde + *n;
	ksteqr_(jobz, n, &w[1], &rwork[inde], &z__[z_offset], ldz, &rwork[
		indrwk], info);
    }

/*     If matrix was scaled, then rescale eigenvalues appropriately. */

    if (iscale == 1) {
	if (*info == 0) {
	    imax = *n;
	} else {
	    imax = *info - 1;
	}
	d__1 = 1. / sigma;
	hscal_(&imax, &d__1, &w[1], &c__1);
    }

    return;

/*     End of ZHBEV */

} /* khbev_ */

