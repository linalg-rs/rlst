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

/* > \brief <b> CHPEV computes the eigenvalues and, optionally, the left and/or right eigenvectors for OTHER m
atrices</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CHPEV + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/chpev.f
"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/chpev.f
"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/chpev.f
"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CHPEV( JOBZ, UPLO, N, AP, W, Z, LDZ, WORK, RWORK, */
/*                         INFO ) */

/*       CHARACTER          JOBZ, UPLO */
/*       INTEGER            INFO, LDZ, N */
/*       REAL               RWORK( * ), W( * ) */
/*       COMPLEX            AP( * ), WORK( * ), Z( LDZ, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CHPEV computes all the eigenvalues and, optionally, eigenvectors of a */
/* > complex Hermitian matrix in packed storage. */
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
/* > \param[in,out] AP */
/* > \verbatim */
/* >          AP is COMPLEX array, dimension (N*(N+1)/2) */
/* >          On entry, the upper or lower triangle of the Hermitian matrix */
/* >          A, packed columnwise in a linear array.  The j-th column of A */
/* >          is stored in the array AP as follows: */
/* >          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/* >          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n. */
/* > */
/* >          On exit, AP is overwritten by values generated during the */
/* >          reduction to tridiagonal form.  If UPLO = 'U', the diagonal */
/* >          and first superdiagonal of the tridiagonal matrix T overwrite */
/* >          the corresponding elements of A, and if UPLO = 'L', the */
/* >          diagonal and first subdiagonal of T overwrite the */
/* >          corresponding elements of A. */
/* > \endverbatim */
/* > */
/* > \param[out] W */
/* > \verbatim */
/* >          W is REAL array, dimension (N) */
/* >          If INFO = 0, the eigenvalues in ascending order. */
/* > \endverbatim */
/* > */
/* > \param[out] Z */
/* > \verbatim */
/* >          Z is COMPLEX array, dimension (LDZ, N) */
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
/* >          WORK is COMPLEX array, dimension (f2cmax(1, 2*N-1)) */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is REAL array, dimension (f2cmax(1, 3*N-2)) */
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

/* > \ingroup complexOTHEReigen */

/*  ===================================================================== */
void  chpev_(char *jobz, char *uplo, integer *n, complex *ap, 
	real *w, complex *z__, integer *ldz, complex *work, real *rwork, 
	integer *info)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1;
    real r__1;

    /* Local variables */
    real eps;
    integer inde;
    real anrm;
    integer imax;
    real rmin, rmax, sigma;
    extern logical lsame_(char *, char *);
    integer iinfo;
    extern void  sscal_(integer *, real *, real *, integer *);
    logical wantz;
    integer iscale;
    extern real clanhp_(char *, char *, integer *, complex *, real *), slamch_(char *);
    extern void  csscal_(integer *, real *, complex *, integer 
	    *);
    real safmin;
    extern void  xerbla_(char *, integer *);
    real bignum;
    integer indtau;
    extern void  chptrd_(char *, integer *, complex *, real *, 
	    real *, complex *, integer *);
    integer indrwk, indwrk;
    extern void  csteqr_(char *, integer *, real *, real *, 
	    complex *, integer *, real *, integer *), cupgtr_(char *, 
	    integer *, complex *, complex *, complex *, integer *, complex *, 
	    integer *), ssterf_(integer *, real *, real *, integer *);
    real smlnum;


/*  -- LAPACK driver routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    --ap;
    --w;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;
    --rwork;

    /* Function Body */
    wantz = lsame_(jobz, "V");

    *info = 0;
    if (! (wantz || lsame_(jobz, "N"))) {
	*info = -1;
    } else if (! (lsame_(uplo, "L") || lsame_(uplo, 
	    "U"))) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*ldz < 1 || wantz && *ldz < *n) {
	*info = -7;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CHPEV ", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

    if (*n == 1) {
	w[1] = ap[1].r;
	rwork[1] = 1.f;
	if (wantz) {
	    i__1 = z_dim1 + 1;
	    z__[i__1].r = 1.f, z__[i__1].i = 0.f;
	}
	return;
    }

/*     Get machine constants. */

    safmin = slamch_("Safe minimum");
    eps = slamch_("Precision");
    smlnum = safmin / eps;
    bignum = 1.f / smlnum;
    rmin = M(sqrt)(smlnum);
    rmax = M(sqrt)(bignum);

/*     Scale matrix to allowable range, if necessary. */

    anrm = clanhp_("M", uplo, n, &ap[1], &rwork[1]);
    iscale = 0;
    if (anrm > 0.f && anrm < rmin) {
	iscale = 1;
	sigma = rmin / anrm;
    } else if (anrm > rmax) {
	iscale = 1;
	sigma = rmax / anrm;
    }
    if (iscale == 1) {
	i__1 = *n * (*n + 1) / 2;
	csscal_(&i__1, &sigma, &ap[1], &c__1);
    }

/*     Call CHPTRD to reduce Hermitian packed matrix to tridiagonal form. */

    inde = 1;
    indtau = 1;
    chptrd_(uplo, n, &ap[1], &w[1], &rwork[inde], &work[indtau], &iinfo);

/*     For eigenvalues only, call SSTERF.  For eigenvectors, first call */
/*     CUPGTR to generate the orthogonal matrix, then call CSTEQR. */

    if (! wantz) {
	ssterf_(n, &w[1], &rwork[inde], info);
    } else {
	indwrk = indtau + *n;
	cupgtr_(uplo, n, &ap[1], &work[indtau], &z__[z_offset], ldz, &work[
		indwrk], &iinfo);
	indrwk = inde + *n;
	csteqr_(jobz, n, &w[1], &rwork[inde], &z__[z_offset], ldz, &rwork[
		indrwk], info);
    }

/*     If matrix was scaled, then rescale eigenvalues appropriately. */

    if (iscale == 1) {
	if (*info == 0) {
	    imax = *n;
	} else {
	    imax = *info - 1;
	}
	r__1 = 1.f / sigma;
	sscal_(&imax, &r__1, &w[1], &c__1);
    }

    return;

/*     End of CHPEV */

} /* chpev_ */

