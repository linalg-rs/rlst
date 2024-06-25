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

static integer c_n1 = -1;
static integer c__1 = 1;
static integer c__0 = 0;
static halfreal c_b36 = 0.;
static halfreal c_b37 = 1.;

/* > \brief <b> DGGES3 computes the eigenvalues, the Schur form, and, optionally, the matrix of Schur vectors 
for GE matrices (blocked algorithm)</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DGGES3 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgges3.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgges3.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgges3.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DGGES3( JOBVSL, JOBVSR, SORT, SELCTG, N, A, LDA, B, LDB, */
/*                          SDIM, ALPHAR, ALPHAI, BETA, VSL, LDVSL, VSR, */
/*                          LDVSR, WORK, LWORK, BWORK, INFO ) */

/*       CHARACTER          JOBVSL, JOBVSR, SORT */
/*       INTEGER            INFO, LDA, LDB, LDVSL, LDVSR, LWORK, N, SDIM */
/*       LOGICAL            BWORK( * ) */
/*       DOUBLE PRECISION   A( LDA, * ), ALPHAI( * ), ALPHAR( * ), */
/*      $                   B( LDB, * ), BETA( * ), VSL( LDVSL, * ), */
/*      $                   VSR( LDVSR, * ), WORK( * ) */
/*       LOGICAL            SELCTG */
/*       EXTERNAL           SELCTG */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DGGES3 computes for a pair of N-by-N doublereal nonsymmetric matrices (A,B), */
/* > the generalized eigenvalues, the generalized doublereal Schur form (S,T), */
/* > optionally, the left and/or right matrices of Schur vectors (VSL and */
/* > VSR). This gives the generalized Schur factorization */
/* > */
/* >          (A,B) = ( (VSL)*S*(VSR)**T, (VSL)*T*(VSR)**T ) */
/* > */
/* > Optionally, it also orders the eigenvalues so that a selected cluster */
/* > of eigenvalues appears in the leading diagonal blocks of the upper */
/* > quasi-triangular matrix S and the upper triangular matrix T.The */
/* > leading columns of VSL and VSR then form an orthonormal basis for the */
/* > corresponding left and right eigenspaces (deflating subspaces). */
/* > */
/* > (If only the generalized eigenvalues are needed, use the driver */
/* > DGGEV instead, which is faster.) */
/* > */
/* > A generalized eigenvalue for a pair of matrices (A,B) is a scalar w */
/* > or a ratio alpha/beta = w, such that  A - w*B is singular.  It is */
/* > usually represented as the pair (alpha,beta), as there is a */
/* > reasonable interpretation for beta=0 or both being zero. */
/* > */
/* > A pair of matrices (S,T) is in generalized doublereal Schur form if T is */
/* > upper triangular with non-negative diagonal and S is block upper */
/* > triangular with 1-by-1 and 2-by-2 blocks.  1-by-1 blocks correspond */
/* > to doublereal generalized eigenvalues, while 2-by-2 blocks of S will be */
/* > "standardized" by making the corresponding elements of T have the */
/* > form: */
/* >         [  a  0  ] */
/* >         [  0  b  ] */
/* > */
/* > and the pair of corresponding 2-by-2 blocks in S and T will have a */
/* > complex conjugate pair of generalized eigenvalues. */
/* > */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] JOBVSL */
/* > \verbatim */
/* >          JOBVSL is CHARACTER*1 */
/* >          = 'N':  do not compute the left Schur vectors; */
/* >          = 'V':  compute the left Schur vectors. */
/* > \endverbatim */
/* > */
/* > \param[in] JOBVSR */
/* > \verbatim */
/* >          JOBVSR is CHARACTER*1 */
/* >          = 'N':  do not compute the right Schur vectors; */
/* >          = 'V':  compute the right Schur vectors. */
/* > \endverbatim */
/* > */
/* > \param[in] SORT */
/* > \verbatim */
/* >          SORT is CHARACTER*1 */
/* >          Specifies whether or not to order the eigenvalues on the */
/* >          diagonal of the generalized Schur form. */
/* >          = 'N':  Eigenvalues are not ordered; */
/* >          = 'S':  Eigenvalues are ordered (see SELCTG); */
/* > \endverbatim */
/* > */
/* > \param[in] SELCTG */
/* > \verbatim */
/* >          SELCTG is a LOGICAL FUNCTION of three DOUBLE PRECISION arguments */
/* >          SELCTG must be declared EXTERNAL in the calling subroutine. */
/* >          If SORT = 'N', SELCTG is not referenced. */
/* >          If SORT = 'S', SELCTG is used to select eigenvalues to sort */
/* >          to the top left of the Schur form. */
/* >          An eigenvalue (ALPHAR(j)+ALPHAI(j))/BETA(j) is selected if */
/* >          SELCTG(ALPHAR(j),ALPHAI(j),BETA(j)) is true; i.e. if either */
/* >          one of a complex conjugate pair of eigenvalues is selected, */
/* >          then both complex eigenvalues are selected. */
/* > */
/* >          Note that in the ill-conditioned case, a selected complex */
/* >          eigenvalue may no longer satisfy SELCTG(ALPHAR(j),ALPHAI(j), */
/* >          BETA(j)) = .TRUE. after ordering. INFO is to be set to N+2 */
/* >          in this case. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrices A, B, VSL, and VSR.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is DOUBLE PRECISION array, dimension (LDA, N) */
/* >          On entry, the first of the pair of matrices. */
/* >          On exit, A has been overwritten by its generalized Schur */
/* >          form S. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is DOUBLE PRECISION array, dimension (LDB, N) */
/* >          On entry, the second of the pair of matrices. */
/* >          On exit, B has been overwritten by its generalized Schur */
/* >          form T. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of B.  LDB >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] SDIM */
/* > \verbatim */
/* >          SDIM is INTEGER */
/* >          If SORT = 'N', SDIM = 0. */
/* >          If SORT = 'S', SDIM = number of eigenvalues (after sorting) */
/* >          for which SELCTG is true.  (Complex conjugate pairs for which */
/* >          SELCTG is true for either eigenvalue count as 2.) */
/* > \endverbatim */
/* > */
/* > \param[out] ALPHAR */
/* > \verbatim */
/* >          ALPHAR is DOUBLE PRECISION array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] ALPHAI */
/* > \verbatim */
/* >          ALPHAI is DOUBLE PRECISION array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] BETA */
/* > \verbatim */
/* >          BETA is DOUBLE PRECISION array, dimension (N) */
/* >          On exit, (ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N, will */
/* >          be the generalized eigenvalues.  ALPHAR(j) + ALPHAI(j)*i, */
/* >          and  BETA(j),j=1,...,N are the diagonals of the complex Schur */
/* >          form (S,T) that would result if the 2-by-2 diagonal blocks of */
/* >          the doublereal Schur form of (A,B) were further reduced to */
/* >          triangular form using 2-by-2 complex unitary transformations. */
/* >          If ALPHAI(j) is zero, then the j-th eigenvalue is doublereal; if */
/* >          positive, then the j-th and (j+1)-st eigenvalues are a */
/* >          complex conjugate pair, with ALPHAI(j+1) negative. */
/* > */
/* >          Note: the quotients ALPHAR(j)/BETA(j) and ALPHAI(j)/BETA(j) */
/* >          may easily over- or underflow, and BETA(j) may even be zero. */
/* >          Thus, the user should avoid naively computing the ratio. */
/* >          However, ALPHAR and ALPHAI will be always less than and */
/* >          usually comparable with norm(A) in magnitude, and BETA always */
/* >          less than and usually comparable with norm(B). */
/* > \endverbatim */
/* > */
/* > \param[out] VSL */
/* > \verbatim */
/* >          VSL is DOUBLE PRECISION array, dimension (LDVSL,N) */
/* >          If JOBVSL = 'V', VSL will contain the left Schur vectors. */
/* >          Not referenced if JOBVSL = 'N'. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVSL */
/* > \verbatim */
/* >          LDVSL is INTEGER */
/* >          The leading dimension of the matrix VSL. LDVSL >=1, and */
/* >          if JOBVSL = 'V', LDVSL >= N. */
/* > \endverbatim */
/* > */
/* > \param[out] VSR */
/* > \verbatim */
/* >          VSR is DOUBLE PRECISION array, dimension (LDVSR,N) */
/* >          If JOBVSR = 'V', VSR will contain the right Schur vectors. */
/* >          Not referenced if JOBVSR = 'N'. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVSR */
/* > \verbatim */
/* >          LDVSR is INTEGER */
/* >          The leading dimension of the matrix VSR. LDVSR >= 1, and */
/* >          if JOBVSR = 'V', LDVSR >= N. */
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
/* >          The dimension of the array WORK. */
/* > */
/* >          If LWORK = -1, then a workspace query is assumed; the routine */
/* >          only calculates the optimal size of the WORK array, returns */
/* >          this value as the first entry of the WORK array, and no error */
/* >          message related to LWORK is issued by XERBLA. */
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
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/* >          = 1,...,N: */
/* >                The QZ iteration failed.  (A,B) are not in Schur */
/* >                form, but ALPHAR(j), ALPHAI(j), and BETA(j) should */
/* >                be correct for j=INFO+1,...,N. */
/* >          > N:  =N+1: other than QZ iteration failed in DHGEQZ. */
/* >                =N+2: after reordering, roundoff changed values of */
/* >                      some complex eigenvalues so that leading */
/* >                      eigenvalues in the Generalized Schur form no */
/* >                      longer satisfy SELCTG=.TRUE.  This could also */
/* >                      be caused due to scaling. */
/* >                =N+3: reordering failed in DTGSEN. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date January 2015 */

/* > \ingroup doubleGEeigen */

/*  ===================================================================== */
void  hgges3_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, integer *n, halfreal *a, integer *lda, halfreal *b, 
	integer *ldb, integer *sdim, halfreal *alphar, halfreal *alphai, 
	halfreal *beta, halfreal *vsl, integer *ldvsl, halfreal *vsr, 
	integer *ldvsr, halfreal *work, integer *lwork, logical *bwork, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, vsl_dim1, vsl_offset, 
	    vsr_dim1, vsr_offset, i__1, i__2;
    halfreal d__1;

    /* Local variables */
    integer i__, ip;
    halfreal dif[2];
    integer ihi, ilo;
    halfreal eps, anrm, bnrm;
    integer idum[1], ierr, itau, iwrk;
    halfreal pvsl, pvsr;
    extern logical lsame_(char *, char *);
    integer ileft, icols;
    logical cursl, ilvsl, ilvsr;
    extern void  hgghd3_(char *, char *, integer *, integer *, 
	    integer *, halfreal *, integer *, halfreal *, integer *, 
	    halfreal *, integer *, halfreal *, integer *, halfreal *, 
	    integer *, integer *);
    integer irows;
    extern void  hlabad_(halfreal *, halfreal *), hggbak_(
	    char *, char *, integer *, integer *, integer *, halfreal *, 
	    halfreal *, integer *, halfreal *, integer *, integer *), hggbal_(char *, integer *, halfreal *, integer 
	    *, halfreal *, integer *, integer *, integer *, halfreal *, 
	    halfreal *, halfreal *, integer *);
    logical lst2sl;
    extern halfreal hlamch_(char *), hlange_(char *, integer *, 
	    integer *, halfreal *, integer *, halfreal *);
    extern void  hlascl_(char *, integer *, integer *, 
	    halfreal *, halfreal *, integer *, integer *, halfreal *, 
	    integer *, integer *);
    logical ilascl, ilbscl;
    extern void  hgeqrf_(integer *, integer *, halfreal *, 
	    integer *, halfreal *, halfreal *, integer *, integer *), 
	    hlacpy_(char *, integer *, integer *, halfreal *, integer *, 
	    halfreal *, integer *);
    halfreal safmin;
    extern void  hlaset_(char *, integer *, integer *, 
	    halfreal *, halfreal *, halfreal *, integer *);
    halfreal safmax;
    extern void  xerbla_(char *, integer *);
    halfreal bignum;
    extern void  hhgeqz_(char *, char *, char *, integer *, 
	    integer *, integer *, halfreal *, integer *, halfreal *, 
	    integer *, halfreal *, halfreal *, halfreal *, halfreal *,
	     integer *, halfreal *, integer *, halfreal *, integer *, 
	    integer *), htgsen_(integer *, logical *, 
	    logical *, logical *, integer *, halfreal *, integer *, 
	    halfreal *, integer *, halfreal *, halfreal *, halfreal *,
	     halfreal *, integer *, halfreal *, integer *, integer *, 
	    halfreal *, halfreal *, halfreal *, halfreal *, integer *,
	     integer *, integer *, integer *);
    integer ijobvl, iright, ijobvr;
    extern void  horgqr_(integer *, integer *, integer *, 
	    halfreal *, integer *, halfreal *, halfreal *, integer *, 
	    integer *);
    halfreal anrmto, bnrmto;
    logical lastsl;
    extern void  hormqr_(char *, char *, integer *, integer *, 
	    integer *, halfreal *, integer *, halfreal *, halfreal *, 
	    integer *, halfreal *, integer *, integer *);
    halfreal smlnum;
    logical wantst, lquery;
    integer lwkopt;


/*  -- LAPACK driver routine (version 3.6.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     January 2015 */


/*  ===================================================================== */


/*     Decode the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --alphar;
    --alphai;
    --beta;
    vsl_dim1 = *ldvsl;
    vsl_offset = 1 + vsl_dim1;
    vsl -= vsl_offset;
    vsr_dim1 = *ldvsr;
    vsr_offset = 1 + vsr_dim1;
    vsr -= vsr_offset;
    --work;
    --bwork;

    /* Function Body */
    if (lsame_(jobvsl, "N")) {
	ijobvl = 1;
	ilvsl = FALSE_;
    } else if (lsame_(jobvsl, "V")) {
	ijobvl = 2;
	ilvsl = TRUE_;
    } else {
	ijobvl = -1;
	ilvsl = FALSE_;
    }

    if (lsame_(jobvsr, "N")) {
	ijobvr = 1;
	ilvsr = FALSE_;
    } else if (lsame_(jobvsr, "V")) {
	ijobvr = 2;
	ilvsr = TRUE_;
    } else {
	ijobvr = -1;
	ilvsr = FALSE_;
    }

    wantst = lsame_(sort, "S");

/*     Test the input arguments */

    *info = 0;
    lquery = *lwork == -1;
    if (ijobvl <= 0) {
	*info = -1;
    } else if (ijobvr <= 0) {
	*info = -2;
    } else if (! wantst && ! lsame_(sort, "N")) {
	*info = -3;
    } else if (*n < 0) {
	*info = -5;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -7;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -9;
    } else if (*ldvsl < 1 || ilvsl && *ldvsl < *n) {
	*info = -15;
    } else if (*ldvsr < 1 || ilvsr && *ldvsr < *n) {
	*info = -17;
    } else if (*lwork < *n * 6 + 16 && ! lquery) {
	*info = -19;
    }

/*     Compute workspace */

    if (*info == 0) {
	hgeqrf_(n, n, &b[b_offset], ldb, &work[1], &work[1], &c_n1, &ierr);
/* Computing MAX */
	i__1 = *n * 6 + 16, i__2 = *n * 3 + (integer) work[1];
	lwkopt = f2cmax(i__1,i__2);
	hormqr_("L", "T", n, n, n, &b[b_offset], ldb, &work[1], &a[a_offset], 
		lda, &work[1], &c_n1, &ierr);
/* Computing MAX */
	i__1 = lwkopt, i__2 = *n * 3 + (integer) work[1];
	lwkopt = f2cmax(i__1,i__2);
	if (ilvsl) {
	    horgqr_(n, n, n, &vsl[vsl_offset], ldvsl, &work[1], &work[1], &
		    c_n1, &ierr);
/* Computing MAX */
	    i__1 = lwkopt, i__2 = *n * 3 + (integer) work[1];
	    lwkopt = f2cmax(i__1,i__2);
	}
	hgghd3_(jobvsl, jobvsr, n, &c__1, n, &a[a_offset], lda, &b[b_offset], 
		ldb, &vsl[vsl_offset], ldvsl, &vsr[vsr_offset], ldvsr, &work[
		1], &c_n1, &ierr);
/* Computing MAX */
	i__1 = lwkopt, i__2 = *n * 3 + (integer) work[1];
	lwkopt = f2cmax(i__1,i__2);
	hhgeqz_("S", jobvsl, jobvsr, n, &c__1, n, &a[a_offset], lda, &b[
		b_offset], ldb, &alphar[1], &alphai[1], &beta[1], &vsl[
		vsl_offset], ldvsl, &vsr[vsr_offset], ldvsr, &work[1], &c_n1, 
		&ierr);
/* Computing MAX */
	i__1 = lwkopt, i__2 = (*n << 1) + (integer) work[1];
	lwkopt = f2cmax(i__1,i__2);
	if (wantst) {
	    htgsen_(&c__0, &ilvsl, &ilvsr, &bwork[1], n, &a[a_offset], lda, &
		    b[b_offset], ldb, &alphar[1], &alphai[1], &beta[1], &vsl[
		    vsl_offset], ldvsl, &vsr[vsr_offset], ldvsr, sdim, &pvsl, 
		    &pvsr, dif, &work[1], &c_n1, idum, &c__1, &ierr);
/* Computing MAX */
	    i__1 = lwkopt, i__2 = (*n << 1) + (integer) work[1];
	    lwkopt = f2cmax(i__1,i__2);
	}
	work[1] = (halfreal) lwkopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGGES3 ", &i__1);
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

    eps = hlamch_("P");
    safmin = hlamch_("S");
    safmax = 1. / safmin;
    hlabad_(&safmin, &safmax);
    smlnum = M(sqrt)(safmin) / eps;
    bignum = 1. / smlnum;

/*     Scale A if f2cmax element outside range [SMLNUM,BIGNUM] */

    anrm = hlange_("M", n, n, &a[a_offset], lda, &work[1]);
    ilascl = FALSE_;
    if (anrm > 0. && anrm < smlnum) {
	anrmto = smlnum;
	ilascl = TRUE_;
    } else if (anrm > bignum) {
	anrmto = bignum;
	ilascl = TRUE_;
    }
    if (ilascl) {
	hlascl_("G", &c__0, &c__0, &anrm, &anrmto, n, n, &a[a_offset], lda, &
		ierr);
    }

/*     Scale B if f2cmax element outside range [SMLNUM,BIGNUM] */

    bnrm = hlange_("M", n, n, &b[b_offset], ldb, &work[1]);
    ilbscl = FALSE_;
    if (bnrm > 0. && bnrm < smlnum) {
	bnrmto = smlnum;
	ilbscl = TRUE_;
    } else if (bnrm > bignum) {
	bnrmto = bignum;
	ilbscl = TRUE_;
    }
    if (ilbscl) {
	hlascl_("G", &c__0, &c__0, &bnrm, &bnrmto, n, n, &b[b_offset], ldb, &
		ierr);
    }

/*     Permute the matrix to make it more nearly triangular */

    ileft = 1;
    iright = *n + 1;
    iwrk = iright + *n;
    hggbal_("P", n, &a[a_offset], lda, &b[b_offset], ldb, &ilo, &ihi, &work[
	    ileft], &work[iright], &work[iwrk], &ierr);

/*     Reduce B to triangular form (QR decomposition of B) */

    irows = ihi + 1 - ilo;
    icols = *n + 1 - ilo;
    itau = iwrk;
    iwrk = itau + irows;
    i__1 = *lwork + 1 - iwrk;
    hgeqrf_(&irows, &icols, &b[ilo + ilo * b_dim1], ldb, &work[itau], &work[
	    iwrk], &i__1, &ierr);

/*     Apply the orthogonal transformation to matrix A */

    i__1 = *lwork + 1 - iwrk;
    hormqr_("L", "T", &irows, &icols, &irows, &b[ilo + ilo * b_dim1], ldb, &
	    work[itau], &a[ilo + ilo * a_dim1], lda, &work[iwrk], &i__1, &
	    ierr);

/*     Initialize VSL */

    if (ilvsl) {
	hlaset_("Full", n, n, &c_b36, &c_b37, &vsl[vsl_offset], ldvsl);
	if (irows > 1) {
	    i__1 = irows - 1;
	    i__2 = irows - 1;
	    hlacpy_("L", &i__1, &i__2, &b[ilo + 1 + ilo * b_dim1], ldb, &vsl[
		    ilo + 1 + ilo * vsl_dim1], ldvsl);
	}
	i__1 = *lwork + 1 - iwrk;
	horgqr_(&irows, &irows, &irows, &vsl[ilo + ilo * vsl_dim1], ldvsl, &
		work[itau], &work[iwrk], &i__1, &ierr);
    }

/*     Initialize VSR */

    if (ilvsr) {
	hlaset_("Full", n, n, &c_b36, &c_b37, &vsr[vsr_offset], ldvsr);
    }

/*     Reduce to generalized Hessenberg form */

    i__1 = *lwork + 1 - iwrk;
    hgghd3_(jobvsl, jobvsr, n, &ilo, &ihi, &a[a_offset], lda, &b[b_offset], 
	    ldb, &vsl[vsl_offset], ldvsl, &vsr[vsr_offset], ldvsr, &work[iwrk]
	    , &i__1, &ierr);

/*     Perform QZ algorithm, computing Schur vectors if desired */

    iwrk = itau;
    i__1 = *lwork + 1 - iwrk;
    hhgeqz_("S", jobvsl, jobvsr, n, &ilo, &ihi, &a[a_offset], lda, &b[
	    b_offset], ldb, &alphar[1], &alphai[1], &beta[1], &vsl[vsl_offset]
	    , ldvsl, &vsr[vsr_offset], ldvsr, &work[iwrk], &i__1, &ierr);
    if (ierr != 0) {
	if (ierr > 0 && ierr <= *n) {
	    *info = ierr;
	} else if (ierr > *n && ierr <= *n << 1) {
	    *info = ierr - *n;
	} else {
	    *info = *n + 1;
	}
	goto L50;
    }

/*     Sort eigenvalues ALPHA/BETA if desired */

    *sdim = 0;
    if (wantst) {

/*        Undo scaling on eigenvalues before SELCTGing */

	if (ilascl) {
	    hlascl_("G", &c__0, &c__0, &anrmto, &anrm, n, &c__1, &alphar[1], 
		    n, &ierr);
	    hlascl_("G", &c__0, &c__0, &anrmto, &anrm, n, &c__1, &alphai[1], 
		    n, &ierr);
	}
	if (ilbscl) {
	    hlascl_("G", &c__0, &c__0, &bnrmto, &bnrm, n, &c__1, &beta[1], n, 
		    &ierr);
	}

/*        Select eigenvalues */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    bwork[i__] = (*selctg)(&alphar[i__], &alphai[i__], &beta[i__]);
/* L10: */
	}

	i__1 = *lwork - iwrk + 1;
	htgsen_(&c__0, &ilvsl, &ilvsr, &bwork[1], n, &a[a_offset], lda, &b[
		b_offset], ldb, &alphar[1], &alphai[1], &beta[1], &vsl[
		vsl_offset], ldvsl, &vsr[vsr_offset], ldvsr, sdim, &pvsl, &
		pvsr, dif, &work[iwrk], &i__1, idum, &c__1, &ierr);
	if (ierr == 1) {
	    *info = *n + 3;
	}

    }

/*     Apply back-permutation to VSL and VSR */

    if (ilvsl) {
	hggbak_("P", "L", n, &ilo, &ihi, &work[ileft], &work[iright], n, &vsl[
		vsl_offset], ldvsl, &ierr);
    }

    if (ilvsr) {
	hggbak_("P", "R", n, &ilo, &ihi, &work[ileft], &work[iright], n, &vsr[
		vsr_offset], ldvsr, &ierr);
    }

/*     Check if unscaling would cause over/underflow, if so, rescale */
/*     (ALPHAR(I),ALPHAI(I),BETA(I)) so BETA(I) is on the order of */
/*     B(I,I) and ALPHAR(I) and ALPHAI(I) are on the order of A(I,I) */

    if (ilascl) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (alphai[i__] != 0.) {
		if (alphar[i__] / safmax > anrmto / anrm || safmin / alphar[
			i__] > anrm / anrmto) {
		    work[1] = (d__1 = a[i__ + i__ * a_dim1] / alphar[i__], 
			    abs(d__1));
		    beta[i__] *= work[1];
		    alphar[i__] *= work[1];
		    alphai[i__] *= work[1];
		} else if (alphai[i__] / safmax > anrmto / anrm || safmin / 
			alphai[i__] > anrm / anrmto) {
		    work[1] = (d__1 = a[i__ + (i__ + 1) * a_dim1] / alphai[
			    i__], abs(d__1));
		    beta[i__] *= work[1];
		    alphar[i__] *= work[1];
		    alphai[i__] *= work[1];
		}
	    }
/* L20: */
	}
    }

    if (ilbscl) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (alphai[i__] != 0.) {
		if (beta[i__] / safmax > bnrmto / bnrm || safmin / beta[i__] 
			> bnrm / bnrmto) {
		    work[1] = (d__1 = b[i__ + i__ * b_dim1] / beta[i__], abs(
			    d__1));
		    beta[i__] *= work[1];
		    alphar[i__] *= work[1];
		    alphai[i__] *= work[1];
		}
	    }
/* L30: */
	}
    }

/*     Undo scaling */

    if (ilascl) {
	hlascl_("H", &c__0, &c__0, &anrmto, &anrm, n, n, &a[a_offset], lda, &
		ierr);
	hlascl_("G", &c__0, &c__0, &anrmto, &anrm, n, &c__1, &alphar[1], n, &
		ierr);
	hlascl_("G", &c__0, &c__0, &anrmto, &anrm, n, &c__1, &alphai[1], n, &
		ierr);
    }

    if (ilbscl) {
	hlascl_("U", &c__0, &c__0, &bnrmto, &bnrm, n, n, &b[b_offset], ldb, &
		ierr);
	hlascl_("G", &c__0, &c__0, &bnrmto, &bnrm, n, &c__1, &beta[1], n, &
		ierr);
    }

    if (wantst) {

/*        Check if reordering is correct */

	lastsl = TRUE_;
	lst2sl = TRUE_;
	*sdim = 0;
	ip = 0;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    cursl = (*selctg)(&alphar[i__], &alphai[i__], &beta[i__]);
	    if (alphai[i__] == 0.) {
		if (cursl) {
		    ++(*sdim);
		}
		ip = 0;
		if (cursl && ! lastsl) {
		    *info = *n + 2;
		}
	    } else {
		if (ip == 1) {

/*                 Last eigenvalue of conjugate pair */

		    cursl = cursl || lastsl;
		    lastsl = cursl;
		    if (cursl) {
			*sdim += 2;
		    }
		    ip = -1;
		    if (cursl && ! lst2sl) {
			*info = *n + 2;
		    }
		} else {

/*                 First eigenvalue of conjugate pair */

		    ip = 1;
		}
	    }
	    lst2sl = lastsl;
	    lastsl = cursl;
/* L40: */
	}

    }

L50:

    work[1] = (halfreal) lwkopt;

    return;

/*     End of DGGES3 */

} /* hgges3_ */

