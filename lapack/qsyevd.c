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
static integer c_n1 = -1;
static integer c__0 = 0;
static quadreal c_b17 = 1.;

/* > \brief <b> DSYEVD computes the eigenvalues and, optionally, the left and/or right eigenvectors for SY mat
rices</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DSYEVD + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dsyevd.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dsyevd.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dsyevd.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DSYEVD( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, IWORK, */
/*                          LIWORK, INFO ) */

/*       CHARACTER          JOBZ, UPLO */
/*       INTEGER            INFO, LDA, LIWORK, LWORK, N */
/*       INTEGER            IWORK( * ) */
/*       DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DSYEVD computes all eigenvalues and, optionally, eigenvectors of a */
/* > doublereal symmetric matrix A. If eigenvectors are desired, it uses a */
/* > divide and conquer algorithm. */
/* > */
/* > The divide and conquer algorithm makes very mild assumptions about */
/* > floating point arithmetic. It will work on machines with a guard */
/* > digit in add/subtract, or on those binary machines without guard */
/* > digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or */
/* > Cray-2. It could conceivably fail on hexadecimal or decimal machines */
/* > without guard digits, but we know of none. */
/* > */
/* > Because of large use of BLAS of level 3, DSYEVD needs N**2 more */
/* > workspace than DSYEVX. */
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
/* >          WORK is DOUBLE PRECISION array, */
/* >                                         dimension (LWORK) */
/* >          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          The dimension of the array WORK. */
/* >          If N <= 1,               LWORK must be at least 1. */
/* >          If JOBZ = 'N' and N > 1, LWORK must be at least 2*N+1. */
/* >          If JOBZ = 'V' and N > 1, LWORK must be at least */
/* >                                                1 + 6*N + 2*N**2. */
/* > */
/* >          If LWORK = -1, then a workspace query is assumed; the routine */
/* >          only calculates the optimal sizes of the WORK and IWORK */
/* >          arrays, returns these values as the first entries of the WORK */
/* >          and IWORK arrays, and no error message related to LWORK or */
/* >          LIWORK is issued by XERBLA. */
/* > \endverbatim */
/* > */
/* > \param[out] IWORK */
/* > \verbatim */
/* >          IWORK is INTEGER array, dimension (MAX(1,LIWORK)) */
/* >          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK. */
/* > \endverbatim */
/* > */
/* > \param[in] LIWORK */
/* > \verbatim */
/* >          LIWORK is INTEGER */
/* >          The dimension of the array IWORK. */
/* >          If N <= 1,                LIWORK must be at least 1. */
/* >          If JOBZ  = 'N' and N > 1, LIWORK must be at least 1. */
/* >          If JOBZ  = 'V' and N > 1, LIWORK must be at least 3 + 5*N. */
/* > */
/* >          If LIWORK = -1, then a workspace query is assumed; the */
/* >          routine only calculates the optimal sizes of the WORK and */
/* >          IWORK arrays, returns these values as the first entries of */
/* >          the WORK and IWORK arrays, and no error message related to */
/* >          LWORK or LIWORK is issued by XERBLA. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  if INFO = i and JOBZ = 'N', then the algorithm failed */
/* >                to converge; i off-diagonal elements of an intermediate */
/* >                tridiagonal form did not converge to zero; */
/* >                if INFO = i and JOBZ = 'V', then the algorithm failed */
/* >                to compute an eigenvalue while working on the submatrix */
/* >                lying in rows and columns INFO/(N+1) through */
/* >                mod(INFO,N+1). */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup doubleSYeigen */

/* > \par Contributors: */
/*  ================== */
/* > */
/* > Jeff Rutter, Computer Science Division, University of California */
/* > at Berkeley, USA \n */
/* >  Modified by Francoise Tisseur, University of Tennessee \n */
/* >  Modified description of INFO. Sven, 16 Feb 05. \n */
/* > */
/*  ===================================================================== */
void  qsyevd_(char *jobz, char *uplo, integer *n, quadreal *
	a, integer *lda, quadreal *w, quadreal *work, integer *lwork, 
	integer *iwork, integer *liwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    quadreal d__1;

    /* Local variables */
    quadreal eps;
    integer inde;
    quadreal anrm, rmin, rmax;
    integer lopt;
    extern void  qscal_(integer *, quadreal *, quadreal *, 
	    integer *);
    quadreal sigma;
    extern logical lsame_(char *, char *);
    integer iinfo, lwmin, liopt;
    logical lower, wantz;
    integer indwk2, llwrk2;
    extern quadreal qlamch_(char *);
    integer iscale;
    extern void  qlascl_(char *, integer *, integer *, 
	    quadreal *, quadreal *, integer *, integer *, quadreal *, 
	    integer *, integer *), qstedc_(char *, integer *, 
	    quadreal *, quadreal *, quadreal *, integer *, quadreal *,
	     integer *, integer *, integer *, integer *), qlacpy_(
	    char *, integer *, integer *, quadreal *, integer *, quadreal 
	    *, integer *);
    quadreal safmin;
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    extern void  xerbla_(char *, integer *);
    quadreal bignum;
    integer indtau;
    extern void  qsterf_(integer *, quadreal *, quadreal *,
	     integer *);
    extern quadreal qlansy_(char *, char *, integer *, quadreal *, 
	    integer *, quadreal *);
    integer indwrk, liwmin;
    extern void  qormtr_(char *, char *, char *, integer *, 
	    integer *, quadreal *, integer *, quadreal *, quadreal *, 
	    integer *, quadreal *, integer *, integer *), qsytrd_(char *, integer *, quadreal *, integer *, 
	    quadreal *, quadreal *, quadreal *, quadreal *, integer *,
	     integer *);
    integer llwork;
    quadreal smlnum;
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
    --iwork;

    /* Function Body */
    wantz = lsame_(jobz, "V");
    lower = lsame_(uplo, "L");
    lquery = *lwork == -1 || *liwork == -1;

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
	if (*n <= 1) {
	    liwmin = 1;
	    lwmin = 1;
	    lopt = lwmin;
	    liopt = liwmin;
	} else {
	    if (wantz) {
		liwmin = *n * 5 + 3;
/* Computing 2nd power */
		i__1 = *n;
		lwmin = *n * 6 + 1 + (i__1 * i__1 << 1);
	    } else {
		liwmin = 1;
		lwmin = (*n << 1) + 1;
	    }
/* Computing MAX */
	    i__1 = lwmin, i__2 = (*n << 1) + ilaenv_(&c__1, "DSYTRD", uplo, n,
		     &c_n1, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
	    lopt = f2cmax(i__1,i__2);
	    liopt = liwmin;
	}
	work[1] = (quadreal) lopt;
	iwork[1] = liopt;

	if (*lwork < lwmin && ! lquery) {
	    *info = -8;
	} else if (*liwork < liwmin && ! lquery) {
	    *info = -10;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSYEVD", &i__1);
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
	if (wantz) {
	    a[a_dim1 + 1] = 1.;
	}
	return;
    }

/*     Get machine constants. */

    safmin = qlamch_("Safe minimum");
    eps = qlamch_("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = M(sqrt)(smlnum);
    rmax = M(sqrt)(bignum);

/*     Scale matrix to allowable range, if necessary. */

    anrm = qlansy_("M", uplo, n, &a[a_offset], lda, &work[1]);
    iscale = 0;
    if (anrm > 0. && anrm < rmin) {
	iscale = 1;
	sigma = rmin / anrm;
    } else if (anrm > rmax) {
	iscale = 1;
	sigma = rmax / anrm;
    }
    if (iscale == 1) {
	qlascl_(uplo, &c__0, &c__0, &c_b17, &sigma, n, n, &a[a_offset], lda, 
		info);
    }

/*     Call DSYTRD to reduce symmetric matrix to tridiagonal form. */

    inde = 1;
    indtau = inde + *n;
    indwrk = indtau + *n;
    llwork = *lwork - indwrk + 1;
    indwk2 = indwrk + *n * *n;
    llwrk2 = *lwork - indwk2 + 1;

    qsytrd_(uplo, n, &a[a_offset], lda, &w[1], &work[inde], &work[indtau], &
	    work[indwrk], &llwork, &iinfo);

/*     For eigenvalues only, call DSTERF.  For eigenvectors, first call */
/*     DSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the */
/*     tridiagonal matrix, then call DORMTR to multiply it by the */
/*     Householder transformations stored in A. */

    if (! wantz) {
	qsterf_(n, &w[1], &work[inde], info);
    } else {
	qstedc_("I", n, &w[1], &work[inde], &work[indwrk], n, &work[indwk2], &
		llwrk2, &iwork[1], liwork, info);
	qormtr_("L", uplo, "N", n, n, &a[a_offset], lda, &work[indtau], &work[
		indwrk], n, &work[indwk2], &llwrk2, &iinfo);
	qlacpy_("A", n, n, &work[indwrk], n, &a[a_offset], lda);
    }

/*     If matrix was scaled, then rescale eigenvalues appropriately. */

    if (iscale == 1) {
	d__1 = 1. / sigma;
	qscal_(n, &d__1, &w[1], &c__1);
    }

    work[1] = (quadreal) lopt;
    iwork[1] = liopt;

    return;

/*     End of DSYEVD */

} /* qsyevd_ */

