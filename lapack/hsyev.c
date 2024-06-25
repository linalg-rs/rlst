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
static integer c_n1 = -1;
static integer c__0 = 0;
static halfreal c_b17 = 1.;

/* > \brief <b> DSYEV computes the eigenvalues and, optionally, the left and/or right eigenvectors for SY matr
ices</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DSYEV + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dsyev.f
"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dsyev.f
"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dsyev.f
"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DSYEV( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, INFO ) */

/*       CHARACTER          JOBZ, UPLO */
/*       INTEGER            INFO, LDA, LWORK, N */
/*       DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DSYEV computes all eigenvalues and, optionally, eigenvectors of a */
/* > doublereal symmetric matrix A. */
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
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is DOUBLE PRECISION array, dimension (LDA, N) */
/* >          On entry, the symmetric matrix A.  If UPLO = 'U', the */
/* >          leading N-by-N upper triangular part of A contains the */
/* >          upper triangular part of the matrix A.  If UPLO = 'L', */
/* >          the leading N-by-N lower triangular part of A contains */
/* >          the lower triangular part of the matrix A. */
/* >          On exit, if JOBZ = 'V', then if INFO = 0, A contains the */
/* >          orthonormal eigenvectors of the matrix A. */
/* >          If JOBZ = 'N', then on exit the lower triangle (if UPLO='L') */
/* >          or the upper triangle (if UPLO='U') of A, including the */
/* >          diagonal, is destroyed. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] W */
/* > \verbatim */
/* >          W is DOUBLE PRECISION array, dimension (N) */
/* >          If INFO = 0, the eigenvalues in ascending order. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK)) */
/* >          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          The length of the array WORK.  LWORK >= f2cmax(1,3*N-1). */
/* >          For optimal efficiency, LWORK >= (NB+2)*N, */
/* >          where NB is the blocksize for DSYTRD returned by ILAENV. */
/* > */
/* >          If LWORK = -1, then a workspace query is assumed; the routine */
/* >          only calculates the optimal size of the WORK array, returns */
/* >          this value as the first entry of the WORK array, and no error */
/* >          message related to LWORK is issued by XERBLA. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
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

/* > \ingroup doubleSYeigen */

/*  ===================================================================== */
void  hsyev_(char *jobz, char *uplo, integer *n, halfreal *a,
	 integer *lda, halfreal *w, halfreal *work, integer *lwork, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    halfreal d__1;

    /* Local variables */
    integer nb;
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
    extern void  hlascl_(char *, integer *, integer *, 
	    halfreal *, halfreal *, integer *, integer *, halfreal *, 
	    integer *, integer *);
    halfreal safmin;
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    extern void  xerbla_(char *, integer *);
    halfreal bignum;
    integer indtau;
    extern void  hsterf_(integer *, halfreal *, halfreal *,
	     integer *);
    extern halfreal hlansy_(char *, char *, integer *, halfreal *, 
	    integer *, halfreal *);
    integer indwrk;
    extern void  horgtr_(char *, integer *, halfreal *, 
	    integer *, halfreal *, halfreal *, integer *, integer *), hsteqr_(char *, integer *, halfreal *, halfreal *, 
	    halfreal *, integer *, halfreal *, integer *), 
	    hsytrd_(char *, integer *, halfreal *, integer *, halfreal *, 
	    halfreal *, halfreal *, halfreal *, integer *, integer *);
    integer llwork;
    halfreal smlnum;
    integer lwkopt;
    logical lquery;


/*  -- LAPACK driver routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --w;
    --work;

    /* Function Body */
    wantz = lsame_(jobz, "V");
    lower = lsame_(uplo, "L");
    lquery = *lwork == -1;

    *info = 0;
    if (! (wantz || lsame_(jobz, "N"))) {
	*info = -1;
    } else if (! (lower || lsame_(uplo, "U"))) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -5;
    }

    if (*info == 0) {
	nb = ilaenv_(&c__1, "DSYTRD", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6,
		 (ftnlen)1);
/* Computing MAX */
	i__1 = 1, i__2 = (nb + 2) * *n;
	lwkopt = f2cmax(i__1,i__2);
	work[1] = (halfreal) lwkopt;

/* Computing MAX */
	i__1 = 1, i__2 = *n * 3 - 1;
	if (*lwork < f2cmax(i__1,i__2) && ! lquery) {
	    *info = -8;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSYEV ", &i__1);
	return;
    } else if (lquery) {
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

    if (*n == 1) {
	w[1] = a[a_dim1 + 1];
	work[1] = 2.;
	if (wantz) {
	    a[a_dim1 + 1] = 1.;
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

    anrm = hlansy_("M", uplo, n, &a[a_offset], lda, &work[1]);
    iscale = 0;
    if (anrm > 0. && anrm < rmin) {
	iscale = 1;
	sigma = rmin / anrm;
    } else if (anrm > rmax) {
	iscale = 1;
	sigma = rmax / anrm;
    }
    if (iscale == 1) {
	hlascl_(uplo, &c__0, &c__0, &c_b17, &sigma, n, n, &a[a_offset], lda, 
		info);
    }

/*     Call DSYTRD to reduce symmetric matrix to tridiagonal form. */

    inde = 1;
    indtau = inde + *n;
    indwrk = indtau + *n;
    llwork = *lwork - indwrk + 1;
    hsytrd_(uplo, n, &a[a_offset], lda, &w[1], &work[inde], &work[indtau], &
	    work[indwrk], &llwork, &iinfo);

/*     For eigenvalues only, call DSTERF.  For eigenvectors, first call */
/*     DORGTR to generate the orthogonal matrix, then call DSTEQR. */

    if (! wantz) {
	hsterf_(n, &w[1], &work[inde], info);
    } else {
	horgtr_(uplo, n, &a[a_offset], lda, &work[indtau], &work[indwrk], &
		llwork, &iinfo);
	hsteqr_(jobz, n, &w[1], &work[inde], &a[a_offset], lda, &work[indtau],
		 info);
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

/*     Set WORK(1) to optimal workspace size. */

    work[1] = (halfreal) lwkopt;

    return;

/*     End of DSYEV */

} /* hsyev_ */

