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
static quadreal c_b42 = 0.;
static quadreal c_b43 = 1.;

/* > \brief <b> DGGESX computes the eigenvalues, the Schur form, and, optionally, the matrix of Schur vectors 
for GE matrices</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DGGESX + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dggesx.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dggesx.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dggesx.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DGGESX( JOBVSL, JOBVSR, SORT, SELCTG, SENSE, N, A, LDA, */
/*                          B, LDB, SDIM, ALPHAR, ALPHAI, BETA, VSL, LDVSL, */
/*                          VSR, LDVSR, RCONDE, RCONDV, WORK, LWORK, IWORK, */
/*                          LIWORK, BWORK, INFO ) */

/*       CHARACTER          JOBVSL, JOBVSR, SENSE, SORT */
/*       INTEGER            INFO, LDA, LDB, LDVSL, LDVSR, LIWORK, LWORK, N, */
/*      $                   SDIM */
/*       LOGICAL            BWORK( * ) */
/*       INTEGER            IWORK( * ) */
/*       DOUBLE PRECISION   A( LDA, * ), ALPHAI( * ), ALPHAR( * ), */
/*      $                   B( LDB, * ), BETA( * ), RCONDE( 2 ), */
/*      $                   RCONDV( 2 ), VSL( LDVSL, * ), VSR( LDVSR, * ), */
/*      $                   WORK( * ) */
/*       LOGICAL            SELCTG */
/*       EXTERNAL           SELCTG */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DGGESX computes for a pair of N-by-N doublereal nonsymmetric matrices */
/* > (A,B), the generalized eigenvalues, the doublereal Schur form (S,T), and, */
/* > optionally, the left and/or right matrices of Schur vectors (VSL and */
/* > VSR).  This gives the generalized Schur factorization */
/* > */
/* >      (A,B) = ( (VSL) S (VSR)**T, (VSL) T (VSR)**T ) */
/* > */
/* > Optionally, it also orders the eigenvalues so that a selected cluster */
/* > of eigenvalues appears in the leading diagonal blocks of the upper */
/* > quasi-triangular matrix S and the upper triangular matrix T; computes */
/* > a reciprocal condition number for the average of the selected */
/* > eigenvalues (RCONDE); and computes a reciprocal condition number for */
/* > the right and left deflating subspaces corresponding to the selected */
/* > eigenvalues (RCONDV). The leading columns of VSL and VSR then form */
/* > an orthonormal basis for the corresponding left and right eigenspaces */
/* > (deflating subspaces). */
/* > */
/* > A generalized eigenvalue for a pair of matrices (A,B) is a scalar w */
/* > or a ratio alpha/beta = w, such that  A - w*B is singular.  It is */
/* > usually represented as the pair (alpha,beta), as there is a */
/* > reasonable interpretation for beta=0 or for both being zero. */
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
/* >          = 'S':  Eigenvalues are ordered (see SELCTG). */
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
/* >          Note that a selected complex eigenvalue may no longer satisfy */
/* >          SELCTG(ALPHAR(j),ALPHAI(j),BETA(j)) = .TRUE. after ordering, */
/* >          since ordering may change the value of complex eigenvalues */
/* >          (especially if the eigenvalue is ill-conditioned), in this */
/* >          case INFO is set to N+3. */
/* > \endverbatim */
/* > */
/* > \param[in] SENSE */
/* > \verbatim */
/* >          SENSE is CHARACTER*1 */
/* >          Determines which reciprocal condition numbers are computed. */
/* >          = 'N' : None are computed; */
/* >          = 'E' : Computed for average of selected eigenvalues only; */
/* >          = 'V' : Computed for selected deflating subspaces only; */
/* >          = 'B' : Computed for both. */
/* >          If SENSE = 'E', 'V', or 'B', SORT must equal 'S'. */
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
/* >          be the generalized eigenvalues.  ALPHAR(j) + ALPHAI(j)*i */
/* >          and BETA(j),j=1,...,N  are the diagonals of the complex Schur */
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
/* > \param[out] RCONDE */
/* > \verbatim */
/* >          RCONDE is DOUBLE PRECISION array, dimension ( 2 ) */
/* >          If SENSE = 'E' or 'B', RCONDE(1) and RCONDE(2) contain the */
/* >          reciprocal condition numbers for the average of the selected */
/* >          eigenvalues. */
/* >          Not referenced if SENSE = 'N' or 'V'. */
/* > \endverbatim */
/* > */
/* > \param[out] RCONDV */
/* > \verbatim */
/* >          RCONDV is DOUBLE PRECISION array, dimension ( 2 ) */
/* >          If SENSE = 'V' or 'B', RCONDV(1) and RCONDV(2) contain the */
/* >          reciprocal condition numbers for the selected deflating */
/* >          subspaces. */
/* >          Not referenced if SENSE = 'N' or 'E'. */
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
/* >          If N = 0, LWORK >= 1, else if SENSE = 'E', 'V', or 'B', */
/* >          LWORK >= f2cmax( 8*N, 6*N+16, 2*SDIM*(N-SDIM) ), else */
/* >          LWORK >= f2cmax( 8*N, 6*N+16 ). */
/* >          Note that 2*SDIM*(N-SDIM) <= N*N/2. */
/* >          Note also that an error is only returned if */
/* >          LWORK < f2cmax( 8*N, 6*N+16), but if SENSE = 'E' or 'V' or 'B' */
/* >          this may not be large enough. */
/* > */
/* >          If LWORK = -1, then a workspace query is assumed; the routine */
/* >          only calculates the bound on the optimal size of the WORK */
/* >          array and the minimum size of the IWORK array, returns these */
/* >          values as the first entries of the WORK and IWORK arrays, and */
/* >          no error message related to LWORK or LIWORK is issued by */
/* >          XERBLA. */
/* > \endverbatim */
/* > */
/* > \param[out] IWORK */
/* > \verbatim */
/* >          IWORK is INTEGER array, dimension (MAX(1,LIWORK)) */
/* >          On exit, if INFO = 0, IWORK(1) returns the minimum LIWORK. */
/* > \endverbatim */
/* > */
/* > \param[in] LIWORK */
/* > \verbatim */
/* >          LIWORK is INTEGER */
/* >          The dimension of the array IWORK. */
/* >          If SENSE = 'N' or N = 0, LIWORK >= 1, otherwise */
/* >          LIWORK >= N+6. */
/* > */
/* >          If LIWORK = -1, then a workspace query is assumed; the */
/* >          routine only calculates the bound on the optimal size of the */
/* >          WORK array and the minimum size of the IWORK array, returns */
/* >          these values as the first entries of the WORK and IWORK */
/* >          arrays, and no error message related to LWORK or LIWORK is */
/* >          issued by XERBLA. */
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
/* >          > N:  =N+1: other than QZ iteration failed in DHGEQZ */
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

/* > \date June 2017 */

/* > \ingroup doubleGEeigen */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  An approximate (asymptotic) bound on the average absolute error of */
/* >  the selected eigenvalues is */
/* > */
/* >       EPS * norm((A, B)) / RCONDE( 1 ). */
/* > */
/* >  An approximate (asymptotic) bound on the maximum angular error in */
/* >  the computed deflating subspaces is */
/* > */
/* >       EPS * norm((A, B)) / RCONDV( 2 ). */
/* > */
/* >  See LAPACK User's Guide, section 4.11 for more information. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  qggesx_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, char *sense, integer *n, quadreal *a, integer *lda, 
	quadreal *b, integer *ldb, integer *sdim, quadreal *alphar, 
	quadreal *alphai, quadreal *beta, quadreal *vsl, integer *ldvsl,
	 quadreal *vsr, integer *ldvsr, quadreal *rconde, quadreal *
	rcondv, quadreal *work, integer *lwork, integer *iwork, integer *
	liwork, logical *bwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, vsl_dim1, vsl_offset, 
	    vsr_dim1, vsr_offset, i__1, i__2;
    quadreal d__1;

    /* Local variables */
    integer i__, ip;
    quadreal pl, pr, dif[2];
    integer ihi, ilo;
    quadreal eps;
    integer ijob;
    quadreal anrm, bnrm;
    integer ierr, itau, iwrk, lwrk;
    extern logical lsame_(char *, char *);
    integer ileft, icols;
    logical cursl, ilvsl, ilvsr;
    integer irows;
    extern void  qlabad_(quadreal *, quadreal *), qggbak_(
	    char *, char *, integer *, integer *, integer *, quadreal *, 
	    quadreal *, integer *, quadreal *, integer *, integer *), qggbal_(char *, integer *, quadreal *, integer 
	    *, quadreal *, integer *, integer *, integer *, quadreal *, 
	    quadreal *, quadreal *, integer *);
    logical lst2sl;
    extern quadreal qlamch_(char *), qlange_(char *, integer *, 
	    integer *, quadreal *, integer *, quadreal *);
    extern void  qgghrd_(char *, char *, integer *, integer *, 
	    integer *, quadreal *, integer *, quadreal *, integer *, 
	    quadreal *, integer *, quadreal *, integer *, integer *), qlascl_(char *, integer *, integer *, quadreal 
	    *, quadreal *, integer *, integer *, quadreal *, integer *, 
	    integer *);
    logical ilascl, ilbscl;
    extern void  qgeqrf_(integer *, integer *, quadreal *, 
	    integer *, quadreal *, quadreal *, integer *, integer *), 
	    qlacpy_(char *, integer *, integer *, quadreal *, integer *, 
	    quadreal *, integer *);
    quadreal safmin;
    extern void  qlaset_(char *, integer *, integer *, 
	    quadreal *, quadreal *, quadreal *, integer *);
    quadreal safmax;
    extern void  xerbla_(char *, integer *);
    quadreal bignum;
    extern void  qhgeqz_(char *, char *, char *, integer *, 
	    integer *, integer *, quadreal *, integer *, quadreal *, 
	    integer *, quadreal *, quadreal *, quadreal *, quadreal *,
	     integer *, quadreal *, integer *, quadreal *, integer *, 
	    integer *);
    integer ijobvl, iright;
    extern void  qtgsen_(integer *, logical *, logical *, 
	    logical *, integer *, quadreal *, integer *, quadreal *, 
	    integer *, quadreal *, quadreal *, quadreal *, quadreal *,
	     integer *, quadreal *, integer *, integer *, quadreal *, 
	    quadreal *, quadreal *, quadreal *, integer *, integer *, 
	    integer *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    integer ijobvr;
    logical wantsb;
    integer liwmin;
    logical wantse, lastsl;
    quadreal anrmto, bnrmto;
    extern void  qorgqr_(integer *, integer *, integer *, 
	    quadreal *, integer *, quadreal *, quadreal *, integer *, 
	    integer *);
    integer minwrk, maxwrk;
    logical wantsn;
    quadreal smlnum;
    extern void  qormqr_(char *, char *, integer *, integer *, 
	    integer *, quadreal *, integer *, quadreal *, quadreal *, 
	    integer *, quadreal *, integer *, integer *);
    logical wantst, lquery, wantsv;


/*  -- LAPACK driver routine (version 3.7.1) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2017 */


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
    --rconde;
    --rcondv;
    --work;
    --iwork;
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
    wantsn = lsame_(sense, "N");
    wantse = lsame_(sense, "E");
    wantsv = lsame_(sense, "V");
    wantsb = lsame_(sense, "B");
    lquery = *lwork == -1 || *liwork == -1;
    if (wantsn) {
	ijob = 0;
    } else if (wantse) {
	ijob = 1;
    } else if (wantsv) {
	ijob = 2;
    } else if (wantsb) {
	ijob = 4;
    }

/*     Test the input arguments */

    *info = 0;
    if (ijobvl <= 0) {
	*info = -1;
    } else if (ijobvr <= 0) {
	*info = -2;
    } else if (! wantst && ! lsame_(sort, "N")) {
	*info = -3;
    } else if (! (wantsn || wantse || wantsv || wantsb) || ! wantst && ! 
	    wantsn) {
	*info = -5;
    } else if (*n < 0) {
	*info = -6;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -8;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -10;
    } else if (*ldvsl < 1 || ilvsl && *ldvsl < *n) {
	*info = -16;
    } else if (*ldvsr < 1 || ilvsr && *ldvsr < *n) {
	*info = -18;
    }

/*     Compute workspace */
/*      (Note: Comments in the code beginning "Workspace:" describe the */
/*       minimal amount of workspace needed at that point in the code, */
/*       as well as the preferred amount for good performance. */
/*       NB refers to the optimal block size for the immediately */
/*       following subroutine, as returned by ILAENV.) */

    if (*info == 0) {
	if (*n > 0) {
/* Computing MAX */
	    i__1 = *n << 3, i__2 = *n * 6 + 16;
	    minwrk = f2cmax(i__1,i__2);
	    maxwrk = minwrk - *n + *n * ilaenv_(&c__1, "DGEQRF", " ", n, &
		    c__1, n, &c__0, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = minwrk - *n + *n * ilaenv_(&c__1, "DORMQR", 
		    " ", n, &c__1, n, &c_n1, (ftnlen)6, (ftnlen)1);
	    maxwrk = f2cmax(i__1,i__2);
	    if (ilvsl) {
/* Computing MAX */
		i__1 = maxwrk, i__2 = minwrk - *n + *n * ilaenv_(&c__1, "DOR"
			"GQR", " ", n, &c__1, n, &c_n1, (ftnlen)6, (ftnlen)1);
		maxwrk = f2cmax(i__1,i__2);
	    }
	    lwrk = maxwrk;
	    if (ijob >= 1) {
/* Computing MAX */
		i__1 = lwrk, i__2 = *n * *n / 2;
		lwrk = f2cmax(i__1,i__2);
	    }
	} else {
	    minwrk = 1;
	    maxwrk = 1;
	    lwrk = 1;
	}
	work[1] = (quadreal) lwrk;
	if (wantsn || *n == 0) {
	    liwmin = 1;
	} else {
	    liwmin = *n + 6;
	}
	iwork[1] = liwmin;

	if (*lwork < minwrk && ! lquery) {
	    *info = -22;
	} else if (*liwork < liwmin && ! lquery) {
	    *info = -24;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGGESX", &i__1);
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
    safmin = qlamch_("S");
    safmax = 1. / safmin;
    qlabad_(&safmin, &safmax);
    smlnum = M(sqrt)(safmin) / eps;
    bignum = 1. / smlnum;

/*     Scale A if f2cmax element outside range [SMLNUM,BIGNUM] */

    anrm = qlange_("M", n, n, &a[a_offset], lda, &work[1]);
    ilascl = FALSE_;
    if (anrm > 0. && anrm < smlnum) {
	anrmto = smlnum;
	ilascl = TRUE_;
    } else if (anrm > bignum) {
	anrmto = bignum;
	ilascl = TRUE_;
    }
    if (ilascl) {
	qlascl_("G", &c__0, &c__0, &anrm, &anrmto, n, n, &a[a_offset], lda, &
		ierr);
    }

/*     Scale B if f2cmax element outside range [SMLNUM,BIGNUM] */

    bnrm = qlange_("M", n, n, &b[b_offset], ldb, &work[1]);
    ilbscl = FALSE_;
    if (bnrm > 0. && bnrm < smlnum) {
	bnrmto = smlnum;
	ilbscl = TRUE_;
    } else if (bnrm > bignum) {
	bnrmto = bignum;
	ilbscl = TRUE_;
    }
    if (ilbscl) {
	qlascl_("G", &c__0, &c__0, &bnrm, &bnrmto, n, n, &b[b_offset], ldb, &
		ierr);
    }

/*     Permute the matrix to make it more nearly triangular */
/*     (Workspace: need 6*N + 2*N for permutation parameters) */

    ileft = 1;
    iright = *n + 1;
    iwrk = iright + *n;
    qggbal_("P", n, &a[a_offset], lda, &b[b_offset], ldb, &ilo, &ihi, &work[
	    ileft], &work[iright], &work[iwrk], &ierr);

/*     Reduce B to triangular form (QR decomposition of B) */
/*     (Workspace: need N, prefer N*NB) */

    irows = ihi + 1 - ilo;
    icols = *n + 1 - ilo;
    itau = iwrk;
    iwrk = itau + irows;
    i__1 = *lwork + 1 - iwrk;
    qgeqrf_(&irows, &icols, &b[ilo + ilo * b_dim1], ldb, &work[itau], &work[
	    iwrk], &i__1, &ierr);

/*     Apply the orthogonal transformation to matrix A */
/*     (Workspace: need N, prefer N*NB) */

    i__1 = *lwork + 1 - iwrk;
    qormqr_("L", "T", &irows, &icols, &irows, &b[ilo + ilo * b_dim1], ldb, &
	    work[itau], &a[ilo + ilo * a_dim1], lda, &work[iwrk], &i__1, &
	    ierr);

/*     Initialize VSL */
/*     (Workspace: need N, prefer N*NB) */

    if (ilvsl) {
	qlaset_("Full", n, n, &c_b42, &c_b43, &vsl[vsl_offset], ldvsl);
	if (irows > 1) {
	    i__1 = irows - 1;
	    i__2 = irows - 1;
	    qlacpy_("L", &i__1, &i__2, &b[ilo + 1 + ilo * b_dim1], ldb, &vsl[
		    ilo + 1 + ilo * vsl_dim1], ldvsl);
	}
	i__1 = *lwork + 1 - iwrk;
	qorgqr_(&irows, &irows, &irows, &vsl[ilo + ilo * vsl_dim1], ldvsl, &
		work[itau], &work[iwrk], &i__1, &ierr);
    }

/*     Initialize VSR */

    if (ilvsr) {
	qlaset_("Full", n, n, &c_b42, &c_b43, &vsr[vsr_offset], ldvsr);
    }

/*     Reduce to generalized Hessenberg form */
/*     (Workspace: none needed) */

    qgghrd_(jobvsl, jobvsr, n, &ilo, &ihi, &a[a_offset], lda, &b[b_offset], 
	    ldb, &vsl[vsl_offset], ldvsl, &vsr[vsr_offset], ldvsr, &ierr);

    *sdim = 0;

/*     Perform QZ algorithm, computing Schur vectors if desired */
/*     (Workspace: need N) */

    iwrk = itau;
    i__1 = *lwork + 1 - iwrk;
    qhgeqz_("S", jobvsl, jobvsr, n, &ilo, &ihi, &a[a_offset], lda, &b[
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
	goto L60;
    }

/*     Sort eigenvalues ALPHA/BETA and compute the reciprocal of */
/*     condition number(s) */
/*     (Workspace: If IJOB >= 1, need MAX( 8*(N+1), 2*SDIM*(N-SDIM) ) */
/*                 otherwise, need 8*(N+1) ) */

    if (wantst) {

/*        Undo scaling on eigenvalues before SELCTGing */

	if (ilascl) {
	    qlascl_("G", &c__0, &c__0, &anrmto, &anrm, n, &c__1, &alphar[1], 
		    n, &ierr);
	    qlascl_("G", &c__0, &c__0, &anrmto, &anrm, n, &c__1, &alphai[1], 
		    n, &ierr);
	}
	if (ilbscl) {
	    qlascl_("G", &c__0, &c__0, &bnrmto, &bnrm, n, &c__1, &beta[1], n, 
		    &ierr);
	}

/*        Select eigenvalues */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    bwork[i__] = (*selctg)(&alphar[i__], &alphai[i__], &beta[i__]);
/* L10: */
	}

/*        Reorder eigenvalues, transform Generalized Schur vectors, and */
/*        compute reciprocal condition numbers */

	i__1 = *lwork - iwrk + 1;
	qtgsen_(&ijob, &ilvsl, &ilvsr, &bwork[1], n, &a[a_offset], lda, &b[
		b_offset], ldb, &alphar[1], &alphai[1], &beta[1], &vsl[
		vsl_offset], ldvsl, &vsr[vsr_offset], ldvsr, sdim, &pl, &pr, 
		dif, &work[iwrk], &i__1, &iwork[1], liwork, &ierr);

	if (ijob >= 1) {
/* Computing MAX */
	    i__1 = maxwrk, i__2 = (*sdim << 1) * (*n - *sdim);
	    maxwrk = f2cmax(i__1,i__2);
	}
	if (ierr == -22) {

/*            not enough doublereal workspace */

	    *info = -22;
	} else {
	    if (ijob == 1 || ijob == 4) {
		rconde[1] = pl;
		rconde[2] = pr;
	    }
	    if (ijob == 2 || ijob == 4) {
		rcondv[1] = dif[0];
		rcondv[2] = dif[1];
	    }
	    if (ierr == 1) {
		*info = *n + 3;
	    }
	}

    }

/*     Apply permutation to VSL and VSR */
/*     (Workspace: none needed) */

    if (ilvsl) {
	qggbak_("P", "L", n, &ilo, &ihi, &work[ileft], &work[iright], n, &vsl[
		vsl_offset], ldvsl, &ierr);
    }

    if (ilvsr) {
	qggbak_("P", "R", n, &ilo, &ihi, &work[ileft], &work[iright], n, &vsr[
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
	qlascl_("H", &c__0, &c__0, &anrmto, &anrm, n, n, &a[a_offset], lda, &
		ierr);
	qlascl_("G", &c__0, &c__0, &anrmto, &anrm, n, &c__1, &alphar[1], n, &
		ierr);
	qlascl_("G", &c__0, &c__0, &anrmto, &anrm, n, &c__1, &alphai[1], n, &
		ierr);
    }

    if (ilbscl) {
	qlascl_("U", &c__0, &c__0, &bnrmto, &bnrm, n, n, &b[b_offset], ldb, &
		ierr);
	qlascl_("G", &c__0, &c__0, &bnrmto, &bnrm, n, &c__1, &beta[1], n, &
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
/* L50: */
	}

    }

L60:

    work[1] = (quadreal) maxwrk;
    iwork[1] = liwmin;

    return;

/*     End of DGGESX */

} /* qggesx_ */

