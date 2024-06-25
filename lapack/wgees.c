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
static integer c__0 = 0;
static integer c_n1 = -1;

/* > \brief <b> ZGEES computes the eigenvalues, the Schur form, and, optionally, the matrix of Schur vectors f
or GE matrices</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZGEES + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zgees.f
"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zgees.f
"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zgees.f
"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZGEES( JOBVS, SORT, SELECT, N, A, LDA, SDIM, W, VS, */
/*                         LDVS, WORK, LWORK, RWORK, BWORK, INFO ) */

/*       CHARACTER          JOBVS, SORT */
/*       INTEGER            INFO, LDA, LDVS, LWORK, N, SDIM */
/*       LOGICAL            BWORK( * ) */
/*       DOUBLE PRECISION   RWORK( * ) */
/*       COMPLEX*16         A( LDA, * ), VS( LDVS, * ), W( * ), WORK( * ) */
/*       LOGICAL            SELECT */
/*       EXTERNAL           SELECT */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZGEES computes for an N-by-N complex nonsymmetric matrix A, the */
/* > eigenvalues, the Schur form T, and, optionally, the matrix of Schur */
/* > vectors Z.  This gives the Schur factorization A = Z*T*(Z**H). */
/* > */
/* > Optionally, it also orders the eigenvalues on the diagonal of the */
/* > Schur form so that selected eigenvalues are at the top left. */
/* > The leading columns of Z then form an orthonormal basis for the */
/* > invariant subspace corresponding to the selected eigenvalues. */
/* > */
/* > A complex matrix is in Schur form if it is upper triangular. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] JOBVS */
/* > \verbatim */
/* >          JOBVS is CHARACTER*1 */
/* >          = 'N': Schur vectors are not computed; */
/* >          = 'V': Schur vectors are computed. */
/* > \endverbatim */
/* > */
/* > \param[in] SORT */
/* > \verbatim */
/* >          SORT is CHARACTER*1 */
/* >          Specifies whether or not to order the eigenvalues on the */
/* >          diagonal of the Schur form. */
/* >          = 'N': Eigenvalues are not ordered: */
/* >          = 'S': Eigenvalues are ordered (see SELECT). */
/* > \endverbatim */
/* > */
/* > \param[in] SELECT */
/* > \verbatim */
/* >          SELECT is a LOGICAL FUNCTION of one COMPLEX*16 argument */
/* >          SELECT must be declared EXTERNAL in the calling subroutine. */
/* >          If SORT = 'S', SELECT is used to select eigenvalues to order */
/* >          to the top left of the Schur form. */
/* >          IF SORT = 'N', SELECT is not referenced. */
/* >          The eigenvalue W(j) is selected if SELECT(W(j)) is true. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A. N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          On entry, the N-by-N matrix A. */
/* >          On exit, A has been overwritten by its Schur form T. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] SDIM */
/* > \verbatim */
/* >          SDIM is INTEGER */
/* >          If SORT = 'N', SDIM = 0. */
/* >          If SORT = 'S', SDIM = number of eigenvalues for which */
/* >                         SELECT is true. */
/* > \endverbatim */
/* > */
/* > \param[out] W */
/* > \verbatim */
/* >          W is COMPLEX*16 array, dimension (N) */
/* >          W contains the computed eigenvalues, in the same order that */
/* >          they appear on the diagonal of the output Schur form T. */
/* > \endverbatim */
/* > */
/* > \param[out] VS */
/* > \verbatim */
/* >          VS is COMPLEX*16 array, dimension (LDVS,N) */
/* >          If JOBVS = 'V', VS contains the unitary matrix Z of Schur */
/* >          vectors. */
/* >          If JOBVS = 'N', VS is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVS */
/* > \verbatim */
/* >          LDVS is INTEGER */
/* >          The leading dimension of the array VS.  LDVS >= 1; if */
/* >          JOBVS = 'V', LDVS >= N. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX*16 array, dimension (MAX(1,LWORK)) */
/* >          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          The dimension of the array WORK.  LWORK >= f2cmax(1,2*N). */
/* >          For good performance, LWORK must generally be larger. */
/* > */
/* >          If LWORK = -1, then a workspace query is assumed; the routine */
/* >          only calculates the optimal size of the WORK array, returns */
/* >          this value as the first entry of the WORK array, and no error */
/* >          message related to LWORK is issued by XERBLA. */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is DOUBLE PRECISION array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] BWORK */
/* > \verbatim */
/* >          BWORK is LOGICAL array, dimension (N) */
/* >          Not referenced if SORT = 'N'. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0: successful exit */
/* >          < 0: if INFO = -i, the i-th argument had an illegal value. */
/* >          > 0: if INFO = i, and i is */
/* >               <= N:  the QR algorithm failed to compute all the */
/* >                      eigenvalues; elements 1:ILO-1 and i+1:N of W */
/* >                      contain those eigenvalues which have converged; */
/* >                      if JOBVS = 'V', VS contains the matrix which */
/* >                      reduces A to its partially converged Schur form. */
/* >               = N+1: the eigenvalues could not be reordered because */
/* >                      some eigenvalues were too close to separate (the */
/* >                      problem is very ill-conditioned); */
/* >               = N+2: after reordering, roundoff changed values of */
/* >                      some complex eigenvalues so that leading */
/* >                      eigenvalues in the Schur form no longer satisfy */
/* >                      SELECT = .TRUE..  This could also be caused by */
/* >                      underflow due to scaling. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16GEeigen */

/*  ===================================================================== */
void  wgees_(char *jobvs, char *sort, L_fp select, integer *n, 
	quadcomplex *a, integer *lda, integer *sdim, quadcomplex *w, 
	quadcomplex *vs, integer *ldvs, quadcomplex *work, integer *lwork,
	 quadreal *rwork, logical *bwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, vs_dim1, vs_offset, i__1, i__2;

    /* Local variables */
    integer i__;
    quadreal s;
    integer ihi, ilo;
    quadreal dum[1], eps, sep;
    integer ibal;
    quadreal anrm;
    integer ierr, itau, iwrk, icond, ieval;
    extern logical lsame_(char *, char *);
    extern void  wcopy_(integer *, quadcomplex *, integer *, 
	    quadcomplex *, integer *), qlabad_(quadreal *, quadreal *);
    logical scalea;
    extern quadreal qlamch_(char *);
    quadreal cscale;
    extern void  wgebak_(char *, char *, integer *, integer *, 
	    integer *, quadreal *, integer *, quadcomplex *, integer *, 
	    integer *), wgebal_(char *, integer *, 
	    quadcomplex *, integer *, integer *, integer *, quadreal *, 
	    integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    extern quadreal wlange_(char *, integer *, integer *, quadcomplex *, 
	    integer *, quadreal *);
    quadreal bignum;
    extern void  wgehrd_(integer *, integer *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, quadcomplex *, 
	    integer *, integer *), wlascl_(char *, integer *, integer *, 
	    quadreal *, quadreal *, integer *, integer *, quadcomplex *,
	     integer *, integer *), wlacpy_(char *, integer *, 
	    integer *, quadcomplex *, integer *, quadcomplex *, integer *);
    integer minwrk, maxwrk;
    quadreal smlnum;
    extern void  whseqr_(char *, char *, integer *, integer *, 
	    integer *, quadcomplex *, integer *, quadcomplex *, 
	    quadcomplex *, integer *, quadcomplex *, integer *, integer *);
    integer hswork;
    extern void  wunghr_(integer *, integer *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, quadcomplex *, 
	    integer *, integer *);
    logical wantst, lquery, wantvs;
    extern void  wtrsen_(char *, char *, logical *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, integer *, 
	    quadcomplex *, integer *, quadreal *, quadreal *, 
	    quadcomplex *, integer *, integer *);


/*  -- LAPACK driver routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --w;
    vs_dim1 = *ldvs;
    vs_offset = 1 + vs_dim1;
    vs -= vs_offset;
    --work;
    --rwork;
    --bwork;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1;
    wantvs = lsame_(jobvs, "V");
    wantst = lsame_(sort, "S");
    if (! wantvs && ! lsame_(jobvs, "N")) {
	*info = -1;
    } else if (! wantst && ! lsame_(sort, "N")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -4;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -6;
    } else if (*ldvs < 1 || wantvs && *ldvs < *n) {
	*info = -10;
    }

/*     Compute workspace */
/*      (Note: Comments in the code beginning "Workspace:" describe the */
/*       minimal amount of workspace needed at that point in the code, */
/*       as well as the preferred amount for good performance. */
/*       CWorkspace refers to complex workspace, and RWorkspace to doublereal */
/*       workspace. NB refers to the optimal block size for the */
/*       immediately following subroutine, as returned by ILAENV. */
/*       HSWORK refers to the workspace preferred by ZHSEQR, as */
/*       calculated below. HSWORK is computed assuming ILO=1 and IHI=N, */
/*       the worst case.) */

    if (*info == 0) {
	if (*n == 0) {
	    minwrk = 1;
	    maxwrk = 1;
	} else {
	    maxwrk = *n + *n * ilaenv_(&c__1, "ZGEHRD", " ", n, &c__1, n, &
		    c__0, (ftnlen)6, (ftnlen)1);
	    minwrk = *n << 1;

	    whseqr_("S", jobvs, n, &c__1, n, &a[a_offset], lda, &w[1], &vs[
		    vs_offset], ldvs, &work[1], &c_n1, &ieval);
	    hswork = (integer) work[1].r;

	    if (! wantvs) {
		maxwrk = f2cmax(maxwrk,hswork);
	    } else {
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n + (*n - 1) * ilaenv_(&c__1, "ZUNGHR",
			 " ", n, &c__1, n, &c_n1, (ftnlen)6, (ftnlen)1);
		maxwrk = f2cmax(i__1,i__2);
		maxwrk = f2cmax(maxwrk,hswork);
	    }
	}
	work[1].r = (quadreal) maxwrk, work[1].i = 0.;

	if (*lwork < minwrk && ! lquery) {
	    *info = -12;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGEES ", &i__1);
	return;
    } else if (lquery) {
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	*sdim = 0;
	return;
    }

/*     Get machine constants */

    eps = qlamch_("P");
    smlnum = qlamch_("S");
    bignum = 1. / smlnum;
    qlabad_(&smlnum, &bignum);
    smlnum = M(sqrt)(smlnum) / eps;
    bignum = 1. / smlnum;

/*     Scale A if f2cmax element outside range [SMLNUM,BIGNUM] */

    anrm = wlange_("M", n, n, &a[a_offset], lda, dum);
    scalea = FALSE_;
    if (anrm > 0. && anrm < smlnum) {
	scalea = TRUE_;
	cscale = smlnum;
    } else if (anrm > bignum) {
	scalea = TRUE_;
	cscale = bignum;
    }
    if (scalea) {
	wlascl_("G", &c__0, &c__0, &anrm, &cscale, n, n, &a[a_offset], lda, &
		ierr);
    }

/*     Permute the matrix to make it more nearly triangular */
/*     (CWorkspace: none) */
/*     (RWorkspace: need N) */

    ibal = 1;
    wgebal_("P", n, &a[a_offset], lda, &ilo, &ihi, &rwork[ibal], &ierr);

/*     Reduce to upper Hessenberg form */
/*     (CWorkspace: need 2*N, prefer N+N*NB) */
/*     (RWorkspace: none) */

    itau = 1;
    iwrk = *n + itau;
    i__1 = *lwork - iwrk + 1;
    wgehrd_(n, &ilo, &ihi, &a[a_offset], lda, &work[itau], &work[iwrk], &i__1,
	     &ierr);

    if (wantvs) {

/*        Copy Householder vectors to VS */

	wlacpy_("L", n, n, &a[a_offset], lda, &vs[vs_offset], ldvs)
		;

/*        Generate unitary matrix in VS */
/*        (CWorkspace: need 2*N-1, prefer N+(N-1)*NB) */
/*        (RWorkspace: none) */

	i__1 = *lwork - iwrk + 1;
	wunghr_(n, &ilo, &ihi, &vs[vs_offset], ldvs, &work[itau], &work[iwrk],
		 &i__1, &ierr);
    }

    *sdim = 0;

/*     Perform QR iteration, accumulating Schur vectors in VS if desired */
/*     (CWorkspace: need 1, prefer HSWORK (see comments) ) */
/*     (RWorkspace: none) */

    iwrk = itau;
    i__1 = *lwork - iwrk + 1;
    whseqr_("S", jobvs, n, &ilo, &ihi, &a[a_offset], lda, &w[1], &vs[
	    vs_offset], ldvs, &work[iwrk], &i__1, &ieval);
    if (ieval > 0) {
	*info = ieval;
    }

/*     Sort eigenvalues if desired */

    if (wantst && *info == 0) {
	if (scalea) {
	    wlascl_("G", &c__0, &c__0, &cscale, &anrm, n, &c__1, &w[1], n, &
		    ierr);
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    bwork[i__] = (*select)(&w[i__]);
/* L10: */
	}

/*        Reorder eigenvalues and transform Schur vectors */
/*        (CWorkspace: none) */
/*        (RWorkspace: none) */

	i__1 = *lwork - iwrk + 1;
	wtrsen_("N", jobvs, &bwork[1], n, &a[a_offset], lda, &vs[vs_offset], 
		ldvs, &w[1], sdim, &s, &sep, &work[iwrk], &i__1, &icond);
    }

    if (wantvs) {

/*        Undo balancing */
/*        (CWorkspace: none) */
/*        (RWorkspace: need N) */

	wgebak_("P", "R", n, &ilo, &ihi, &rwork[ibal], n, &vs[vs_offset], 
		ldvs, &ierr);
    }

    if (scalea) {

/*        Undo scaling for the Schur form of A */

	wlascl_("U", &c__0, &c__0, &cscale, &anrm, n, n, &a[a_offset], lda, &
		ierr);
	i__1 = *lda + 1;
	wcopy_(n, &a[a_offset], &i__1, &w[1], &c__1);
    }

    work[1].r = (quadreal) maxwrk, work[1].i = 0.;
    return;

/*     End of ZGEES */

} /* wgees_ */

