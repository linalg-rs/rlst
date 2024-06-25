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
static integer c__0 = 0;
static halfreal c_b59 = 0.;
static halfreal c_b60 = 1.;

/* > \brief <b> DGGEVX computes the eigenvalues and, optionally, the left and/or right eigenvectors for GE mat
rices</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DGGEVX + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dggevx.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dggevx.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dggevx.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DGGEVX( BALANC, JOBVL, JOBVR, SENSE, N, A, LDA, B, LDB, */
/*                          ALPHAR, ALPHAI, BETA, VL, LDVL, VR, LDVR, ILO, */
/*                          IHI, LSCALE, RSCALE, ABNRM, BBNRM, RCONDE, */
/*                          RCONDV, WORK, LWORK, IWORK, BWORK, INFO ) */

/*       CHARACTER          BALANC, JOBVL, JOBVR, SENSE */
/*       INTEGER            IHI, ILO, INFO, LDA, LDB, LDVL, LDVR, LWORK, N */
/*       DOUBLE PRECISION   ABNRM, BBNRM */
/*       LOGICAL            BWORK( * ) */
/*       INTEGER            IWORK( * ) */
/*       DOUBLE PRECISION   A( LDA, * ), ALPHAI( * ), ALPHAR( * ), */
/*      $                   B( LDB, * ), BETA( * ), LSCALE( * ), */
/*      $                   RCONDE( * ), RCONDV( * ), RSCALE( * ), */
/*      $                   VL( LDVL, * ), VR( LDVR, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DGGEVX computes for a pair of N-by-N doublereal nonsymmetric matrices (A,B) */
/* > the generalized eigenvalues, and optionally, the left and/or right */
/* > generalized eigenvectors. */
/* > */
/* > Optionally also, it computes a balancing transformation to improve */
/* > the conditioning of the eigenvalues and eigenvectors (ILO, IHI, */
/* > LSCALE, RSCALE, ABNRM, and BBNRM), reciprocal condition numbers for */
/* > the eigenvalues (RCONDE), and reciprocal condition numbers for the */
/* > right eigenvectors (RCONDV). */
/* > */
/* > A generalized eigenvalue for a pair of matrices (A,B) is a scalar */
/* > lambda or a ratio alpha/beta = lambda, such that A - lambda*B is */
/* > singular. It is usually represented as the pair (alpha,beta), as */
/* > there is a reasonable interpretation for beta=0, and even for both */
/* > being zero. */
/* > */
/* > The right eigenvector v(j) corresponding to the eigenvalue lambda(j) */
/* > of (A,B) satisfies */
/* > */
/* >                  A * v(j) = lambda(j) * B * v(j) . */
/* > */
/* > The left eigenvector u(j) corresponding to the eigenvalue lambda(j) */
/* > of (A,B) satisfies */
/* > */
/* >                  u(j)**H * A  = lambda(j) * u(j)**H * B. */
/* > */
/* > where u(j)**H is the conjugate-transpose of u(j). */
/* > */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] BALANC */
/* > \verbatim */
/* >          BALANC is CHARACTER*1 */
/* >          Specifies the balance option to be performed. */
/* >          = 'N':  do not diagonally scale or permute; */
/* >          = 'P':  permute only; */
/* >          = 'S':  scale only; */
/* >          = 'B':  both permute and scale. */
/* >          Computed reciprocal condition numbers will be for the */
/* >          matrices after permuting and/or balancing. Permuting does */
/* >          not change condition numbers (in exact arithmetic), but */
/* >          balancing does. */
/* > \endverbatim */
/* > */
/* > \param[in] JOBVL */
/* > \verbatim */
/* >          JOBVL is CHARACTER*1 */
/* >          = 'N':  do not compute the left generalized eigenvectors; */
/* >          = 'V':  compute the left generalized eigenvectors. */
/* > \endverbatim */
/* > */
/* > \param[in] JOBVR */
/* > \verbatim */
/* >          JOBVR is CHARACTER*1 */
/* >          = 'N':  do not compute the right generalized eigenvectors; */
/* >          = 'V':  compute the right generalized eigenvectors. */
/* > \endverbatim */
/* > */
/* > \param[in] SENSE */
/* > \verbatim */
/* >          SENSE is CHARACTER*1 */
/* >          Determines which reciprocal condition numbers are computed. */
/* >          = 'N': none are computed; */
/* >          = 'E': computed for eigenvalues only; */
/* >          = 'V': computed for eigenvectors only; */
/* >          = 'B': computed for eigenvalues and eigenvectors. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrices A, B, VL, and VR.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is DOUBLE PRECISION array, dimension (LDA, N) */
/* >          On entry, the matrix A in the pair (A,B). */
/* >          On exit, A has been overwritten. If JOBVL='V' or JOBVR='V' */
/* >          or both, then A contains the first part of the doublereal Schur */
/* >          form of the "balanced" versions of the input A and B. */
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
/* >          On entry, the matrix B in the pair (A,B). */
/* >          On exit, B has been overwritten. If JOBVL='V' or JOBVR='V' */
/* >          or both, then B contains the second part of the doublereal Schur */
/* >          form of the "balanced" versions of the input A and B. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of B.  LDB >= f2cmax(1,N). */
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
/* >          be the generalized eigenvalues.  If ALPHAI(j) is zero, then */
/* >          the j-th eigenvalue is doublereal; if positive, then the j-th and */
/* >          (j+1)-st eigenvalues are a complex conjugate pair, with */
/* >          ALPHAI(j+1) negative. */
/* > */
/* >          Note: the quotients ALPHAR(j)/BETA(j) and ALPHAI(j)/BETA(j) */
/* >          may easily over- or underflow, and BETA(j) may even be zero. */
/* >          Thus, the user should avoid naively computing the ratio */
/* >          ALPHA/BETA. However, ALPHAR and ALPHAI will be always less */
/* >          than and usually comparable with norm(A) in magnitude, and */
/* >          BETA always less than and usually comparable with norm(B). */
/* > \endverbatim */
/* > */
/* > \param[out] VL */
/* > \verbatim */
/* >          VL is DOUBLE PRECISION array, dimension (LDVL,N) */
/* >          If JOBVL = 'V', the left eigenvectors u(j) are stored one */
/* >          after another in the columns of VL, in the same order as */
/* >          their eigenvalues. If the j-th eigenvalue is doublereal, then */
/* >          u(j) = VL(:,j), the j-th column of VL. If the j-th and */
/* >          (j+1)-th eigenvalues form a complex conjugate pair, then */
/* >          u(j) = VL(:,j)+i*VL(:,j+1) and u(j+1) = VL(:,j)-i*VL(:,j+1). */
/* >          Each eigenvector will be scaled so the largest component have */
/* >          abs(doublereal part) + abs(imag. part) = 1. */
/* >          Not referenced if JOBVL = 'N'. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVL */
/* > \verbatim */
/* >          LDVL is INTEGER */
/* >          The leading dimension of the matrix VL. LDVL >= 1, and */
/* >          if JOBVL = 'V', LDVL >= N. */
/* > \endverbatim */
/* > */
/* > \param[out] VR */
/* > \verbatim */
/* >          VR is DOUBLE PRECISION array, dimension (LDVR,N) */
/* >          If JOBVR = 'V', the right eigenvectors v(j) are stored one */
/* >          after another in the columns of VR, in the same order as */
/* >          their eigenvalues. If the j-th eigenvalue is doublereal, then */
/* >          v(j) = VR(:,j), the j-th column of VR. If the j-th and */
/* >          (j+1)-th eigenvalues form a complex conjugate pair, then */
/* >          v(j) = VR(:,j)+i*VR(:,j+1) and v(j+1) = VR(:,j)-i*VR(:,j+1). */
/* >          Each eigenvector will be scaled so the largest component have */
/* >          abs(doublereal part) + abs(imag. part) = 1. */
/* >          Not referenced if JOBVR = 'N'. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVR */
/* > \verbatim */
/* >          LDVR is INTEGER */
/* >          The leading dimension of the matrix VR. LDVR >= 1, and */
/* >          if JOBVR = 'V', LDVR >= N. */
/* > \endverbatim */
/* > */
/* > \param[out] ILO */
/* > \verbatim */
/* >          ILO is INTEGER */
/* > \endverbatim */
/* > */
/* > \param[out] IHI */
/* > \verbatim */
/* >          IHI is INTEGER */
/* >          ILO and IHI are integer values such that on exit */
/* >          A(i,j) = 0 and B(i,j) = 0 if i > j and */
/* >          j = 1,...,ILO-1 or i = IHI+1,...,N. */
/* >          If BALANC = 'N' or 'S', ILO = 1 and IHI = N. */
/* > \endverbatim */
/* > */
/* > \param[out] LSCALE */
/* > \verbatim */
/* >          LSCALE is DOUBLE PRECISION array, dimension (N) */
/* >          Details of the permutations and scaling factors applied */
/* >          to the left side of A and B.  If PL(j) is the index of the */
/* >          row interchanged with row j, and DL(j) is the scaling */
/* >          factor applied to row j, then */
/* >            LSCALE(j) = PL(j)  for j = 1,...,ILO-1 */
/* >                      = DL(j)  for j = ILO,...,IHI */
/* >                      = PL(j)  for j = IHI+1,...,N. */
/* >          The order in which the interchanges are made is N to IHI+1, */
/* >          then 1 to ILO-1. */
/* > \endverbatim */
/* > */
/* > \param[out] RSCALE */
/* > \verbatim */
/* >          RSCALE is DOUBLE PRECISION array, dimension (N) */
/* >          Details of the permutations and scaling factors applied */
/* >          to the right side of A and B.  If PR(j) is the index of the */
/* >          column interchanged with column j, and DR(j) is the scaling */
/* >          factor applied to column j, then */
/* >            RSCALE(j) = PR(j)  for j = 1,...,ILO-1 */
/* >                      = DR(j)  for j = ILO,...,IHI */
/* >                      = PR(j)  for j = IHI+1,...,N */
/* >          The order in which the interchanges are made is N to IHI+1, */
/* >          then 1 to ILO-1. */
/* > \endverbatim */
/* > */
/* > \param[out] ABNRM */
/* > \verbatim */
/* >          ABNRM is DOUBLE PRECISION */
/* >          The one-norm of the balanced matrix A. */
/* > \endverbatim */
/* > */
/* > \param[out] BBNRM */
/* > \verbatim */
/* >          BBNRM is DOUBLE PRECISION */
/* >          The one-norm of the balanced matrix B. */
/* > \endverbatim */
/* > */
/* > \param[out] RCONDE */
/* > \verbatim */
/* >          RCONDE is DOUBLE PRECISION array, dimension (N) */
/* >          If SENSE = 'E' or 'B', the reciprocal condition numbers of */
/* >          the eigenvalues, stored in consecutive elements of the array. */
/* >          For a complex conjugate pair of eigenvalues two consecutive */
/* >          elements of RCONDE are set to the same value. Thus RCONDE(j), */
/* >          RCONDV(j), and the j-th columns of VL and VR all correspond */
/* >          to the j-th eigenpair. */
/* >          If SENSE = 'N or 'V', RCONDE is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[out] RCONDV */
/* > \verbatim */
/* >          RCONDV is DOUBLE PRECISION array, dimension (N) */
/* >          If SENSE = 'V' or 'B', the estimated reciprocal condition */
/* >          numbers of the eigenvectors, stored in consecutive elements */
/* >          of the array. For a complex eigenvector two consecutive */
/* >          elements of RCONDV are set to the same value. If the */
/* >          eigenvalues cannot be reordered to compute RCONDV(j), */
/* >          RCONDV(j) is set to 0; this can only occur when the true */
/* >          value would be very small anyway. */
/* >          If SENSE = 'N' or 'E', RCONDV is not referenced. */
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
/* >          The dimension of the array WORK. LWORK >= f2cmax(1,2*N). */
/* >          If BALANC = 'S' or 'B', or JOBVL = 'V', or JOBVR = 'V', */
/* >          LWORK >= f2cmax(1,6*N). */
/* >          If SENSE = 'E' or 'B', LWORK >= f2cmax(1,10*N). */
/* >          If SENSE = 'V' or 'B', LWORK >= 2*N*N+8*N+16. */
/* > */
/* >          If LWORK = -1, then a workspace query is assumed; the routine */
/* >          only calculates the optimal size of the WORK array, returns */
/* >          this value as the first entry of the WORK array, and no error */
/* >          message related to LWORK is issued by XERBLA. */
/* > \endverbatim */
/* > */
/* > \param[out] IWORK */
/* > \verbatim */
/* >          IWORK is INTEGER array, dimension (N+6) */
/* >          If SENSE = 'E', IWORK is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[out] BWORK */
/* > \verbatim */
/* >          BWORK is LOGICAL array, dimension (N) */
/* >          If SENSE = 'N', BWORK is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/* >          = 1,...,N: */
/* >                The QZ iteration failed.  No eigenvectors have been */
/* >                calculated, but ALPHAR(j), ALPHAI(j), and BETA(j) */
/* >                should be correct for j=INFO+1,...,N. */
/* >          > N:  =N+1: other than QZ iteration failed in DHGEQZ. */
/* >                =N+2: error return from DTGEVC. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date April 2012 */

/* > \ingroup doubleGEeigen */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  Balancing a matrix pair (A,B) includes, first, permuting rows and */
/* >  columns to isolate eigenvalues, second, applying diagonal similarity */
/* >  transformation to the rows and columns to make the rows and columns */
/* >  as close in norm as possible. The computed reciprocal condition */
/* >  numbers correspond to the balanced matrix. Permuting rows and columns */
/* >  will not change the condition numbers (in exact arithmetic) but */
/* >  diagonal scaling will.  For further explanation of balancing, see */
/* >  section 4.11.1.2 of LAPACK Users' Guide. */
/* > */
/* >  An approximate error bound on the chordal distance between the i-th */
/* >  computed generalized eigenvalue w and the corresponding exact */
/* >  eigenvalue lambda is */
/* > */
/* >       chord(w, lambda) <= EPS * norm(ABNRM, BBNRM) / RCONDE(I) */
/* > */
/* >  An approximate error bound for the angle between the i-th computed */
/* >  eigenvector VL(i) or VR(i) is given by */
/* > */
/* >       EPS * norm(ABNRM, BBNRM) / DIF(i). */
/* > */
/* >  For further explanation of the reciprocal condition numbers RCONDE */
/* >  and RCONDV, see section 4.11 of LAPACK User's Guide. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  hggevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, halfreal *a, integer *lda, halfreal *b, 
	integer *ldb, halfreal *alphar, halfreal *alphai, halfreal *
	beta, halfreal *vl, integer *ldvl, halfreal *vr, integer *ldvr, 
	integer *ilo, integer *ihi, halfreal *lscale, halfreal *rscale, 
	halfreal *abnrm, halfreal *bbnrm, halfreal *rconde, halfreal *
	rcondv, halfreal *work, integer *lwork, integer *iwork, logical *
	bwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, vl_dim1, vl_offset, vr_dim1, 
	    vr_offset, i__1, i__2;
    halfreal d__1, d__2, d__3, d__4;

    /* Local variables */
    integer i__, j, m, jc, in, mm, jr;
    halfreal eps;
    logical ilv, pair;
    halfreal anrm, bnrm;
    integer ierr, itau;
    halfreal temp;
    logical ilvl, ilvr;
    integer iwrk, iwrk1;
    extern logical lsame_(char *, char *);
    integer icols;
    logical noscl;
    integer irows;
    extern void  hlabad_(halfreal *, halfreal *), hggbak_(
	    char *, char *, integer *, integer *, integer *, halfreal *, 
	    halfreal *, integer *, halfreal *, integer *, integer *), hggbal_(char *, integer *, halfreal *, integer 
	    *, halfreal *, integer *, integer *, integer *, halfreal *, 
	    halfreal *, halfreal *, integer *);
    extern halfreal hlamch_(char *), hlange_(char *, integer *, 
	    integer *, halfreal *, integer *, halfreal *);
    extern void  hgghrd_(char *, char *, integer *, integer *, 
	    integer *, halfreal *, integer *, halfreal *, integer *, 
	    halfreal *, integer *, halfreal *, integer *, integer *), hlascl_(char *, integer *, integer *, halfreal 
	    *, halfreal *, integer *, integer *, halfreal *, integer *, 
	    integer *);
    logical ilascl, ilbscl;
    extern void  hgeqrf_(integer *, integer *, halfreal *, 
	    integer *, halfreal *, halfreal *, integer *, integer *), 
	    hlacpy_(char *, integer *, integer *, halfreal *, integer *, 
	    halfreal *, integer *), hlaset_(char *, integer *, 
	    integer *, halfreal *, halfreal *, halfreal *, integer *);
    logical ldumma[1];
    char chtemp[1];
    halfreal bignum;
    extern void  hhgeqz_(char *, char *, char *, integer *, 
	    integer *, integer *, halfreal *, integer *, halfreal *, 
	    integer *, halfreal *, halfreal *, halfreal *, halfreal *,
	     integer *, halfreal *, integer *, halfreal *, integer *, 
	    integer *), htgevc_(char *, char *, 
	    logical *, integer *, halfreal *, integer *, halfreal *, 
	    integer *, halfreal *, integer *, halfreal *, integer *, 
	    integer *, integer *, halfreal *, integer *);
    integer ijobvl;
    extern void  htgsna_(char *, char *, logical *, integer *, 
	    halfreal *, integer *, halfreal *, integer *, halfreal *, 
	    integer *, halfreal *, integer *, halfreal *, halfreal *, 
	    integer *, integer *, halfreal *, integer *, integer *, integer 
	    *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    integer ijobvr;
    logical wantsb;
    extern void  horgqr_(integer *, integer *, integer *, 
	    halfreal *, integer *, halfreal *, halfreal *, integer *, 
	    integer *);
    halfreal anrmto;
    logical wantse;
    halfreal bnrmto;
    extern void  hormqr_(char *, char *, integer *, integer *, 
	    integer *, halfreal *, integer *, halfreal *, halfreal *, 
	    integer *, halfreal *, integer *, integer *);
    integer minwrk, maxwrk;
    logical wantsn;
    halfreal smlnum;
    logical lquery, wantsv;


/*  -- LAPACK driver routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     April 2012 */


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
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1;
    vr -= vr_offset;
    --lscale;
    --rscale;
    --rconde;
    --rcondv;
    --work;
    --iwork;
    --bwork;

    /* Function Body */
    if (lsame_(jobvl, "N")) {
	ijobvl = 1;
	ilvl = FALSE_;
    } else if (lsame_(jobvl, "V")) {
	ijobvl = 2;
	ilvl = TRUE_;
    } else {
	ijobvl = -1;
	ilvl = FALSE_;
    }

    if (lsame_(jobvr, "N")) {
	ijobvr = 1;
	ilvr = FALSE_;
    } else if (lsame_(jobvr, "V")) {
	ijobvr = 2;
	ilvr = TRUE_;
    } else {
	ijobvr = -1;
	ilvr = FALSE_;
    }
    ilv = ilvl || ilvr;

    noscl = lsame_(balanc, "N") || lsame_(balanc, "P");
    wantsn = lsame_(sense, "N");
    wantse = lsame_(sense, "E");
    wantsv = lsame_(sense, "V");
    wantsb = lsame_(sense, "B");

/*     Test the input arguments */

    *info = 0;
    lquery = *lwork == -1;
    if (! (lsame_(balanc, "N") || lsame_(balanc, "S") || lsame_(balanc, "P") 
	    || lsame_(balanc, "B"))) {
	*info = -1;
    } else if (ijobvl <= 0) {
	*info = -2;
    } else if (ijobvr <= 0) {
	*info = -3;
    } else if (! (wantsn || wantse || wantsb || wantsv)) {
	*info = -4;
    } else if (*n < 0) {
	*info = -5;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -7;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -9;
    } else if (*ldvl < 1 || ilvl && *ldvl < *n) {
	*info = -14;
    } else if (*ldvr < 1 || ilvr && *ldvr < *n) {
	*info = -16;
    }

/*     Compute workspace */
/*      (Note: Comments in the code beginning "Workspace:" describe the */
/*       minimal amount of workspace needed at that point in the code, */
/*       as well as the preferred amount for good performance. */
/*       NB refers to the optimal block size for the immediately */
/*       following subroutine, as returned by ILAENV. The workspace is */
/*       computed assuming ILO = 1 and IHI = N, the worst case.) */

    if (*info == 0) {
	if (*n == 0) {
	    minwrk = 1;
	    maxwrk = 1;
	} else {
	    if (noscl && ! ilv) {
		minwrk = *n << 1;
	    } else {
		minwrk = *n * 6;
	    }
	    if (wantse || wantsb) {
		minwrk = *n * 10;
	    }
	    if (wantsv || wantsb) {
/* Computing MAX */
		i__1 = minwrk, i__2 = (*n << 1) * (*n + 4) + 16;
		minwrk = f2cmax(i__1,i__2);
	    }
	    maxwrk = minwrk;
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n + *n * ilaenv_(&c__1, "DGEQRF", " ", n, &
		    c__1, n, &c__0, (ftnlen)6, (ftnlen)1);
	    maxwrk = f2cmax(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n + *n * ilaenv_(&c__1, "DORMQR", " ", n, &
		    c__1, n, &c__0, (ftnlen)6, (ftnlen)1);
	    maxwrk = f2cmax(i__1,i__2);
	    if (ilvl) {
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n + *n * ilaenv_(&c__1, "DORGQR", 
			" ", n, &c__1, n, &c__0, (ftnlen)6, (ftnlen)1);
		maxwrk = f2cmax(i__1,i__2);
	    }
	}
	work[1] = (halfreal) maxwrk;

	if (*lwork < minwrk && ! lquery) {
	    *info = -26;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGGEVX", &i__1);
	return;
    } else if (lquery) {
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }


/*     Get machine constants */

    eps = hlamch_("P");
    smlnum = hlamch_("S");
    bignum = 1. / smlnum;
    hlabad_(&smlnum, &bignum);
    smlnum = M(sqrt)(smlnum) / eps;
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

/*     Permute and/or balance the matrix pair (A,B) */
/*     (Workspace: need 6*N if BALANC = 'S' or 'B', 1 otherwise) */

    hggbal_(balanc, n, &a[a_offset], lda, &b[b_offset], ldb, ilo, ihi, &
	    lscale[1], &rscale[1], &work[1], &ierr);

/*     Compute ABNRM and BBNRM */

    *abnrm = hlange_("1", n, n, &a[a_offset], lda, &work[1]);
    if (ilascl) {
	work[1] = *abnrm;
	hlascl_("G", &c__0, &c__0, &anrmto, &anrm, &c__1, &c__1, &work[1], &
		c__1, &ierr);
	*abnrm = work[1];
    }

    *bbnrm = hlange_("1", n, n, &b[b_offset], ldb, &work[1]);
    if (ilbscl) {
	work[1] = *bbnrm;
	hlascl_("G", &c__0, &c__0, &bnrmto, &bnrm, &c__1, &c__1, &work[1], &
		c__1, &ierr);
	*bbnrm = work[1];
    }

/*     Reduce B to triangular form (QR decomposition of B) */
/*     (Workspace: need N, prefer N*NB ) */

    irows = *ihi + 1 - *ilo;
    if (ilv || ! wantsn) {
	icols = *n + 1 - *ilo;
    } else {
	icols = irows;
    }
    itau = 1;
    iwrk = itau + irows;
    i__1 = *lwork + 1 - iwrk;
    hgeqrf_(&irows, &icols, &b[*ilo + *ilo * b_dim1], ldb, &work[itau], &work[
	    iwrk], &i__1, &ierr);

/*     Apply the orthogonal transformation to A */
/*     (Workspace: need N, prefer N*NB) */

    i__1 = *lwork + 1 - iwrk;
    hormqr_("L", "T", &irows, &icols, &irows, &b[*ilo + *ilo * b_dim1], ldb, &
	    work[itau], &a[*ilo + *ilo * a_dim1], lda, &work[iwrk], &i__1, &
	    ierr);

/*     Initialize VL and/or VR */
/*     (Workspace: need N, prefer N*NB) */

    if (ilvl) {
	hlaset_("Full", n, n, &c_b59, &c_b60, &vl[vl_offset], ldvl)
		;
	if (irows > 1) {
	    i__1 = irows - 1;
	    i__2 = irows - 1;
	    hlacpy_("L", &i__1, &i__2, &b[*ilo + 1 + *ilo * b_dim1], ldb, &vl[
		    *ilo + 1 + *ilo * vl_dim1], ldvl);
	}
	i__1 = *lwork + 1 - iwrk;
	horgqr_(&irows, &irows, &irows, &vl[*ilo + *ilo * vl_dim1], ldvl, &
		work[itau], &work[iwrk], &i__1, &ierr);
    }

    if (ilvr) {
	hlaset_("Full", n, n, &c_b59, &c_b60, &vr[vr_offset], ldvr)
		;
    }

/*     Reduce to generalized Hessenberg form */
/*     (Workspace: none needed) */

    if (ilv || ! wantsn) {

/*        Eigenvectors requested -- work on whole matrix. */

	hgghrd_(jobvl, jobvr, n, ilo, ihi, &a[a_offset], lda, &b[b_offset], 
		ldb, &vl[vl_offset], ldvl, &vr[vr_offset], ldvr, &ierr);
    } else {
	hgghrd_("N", "N", &irows, &c__1, &irows, &a[*ilo + *ilo * a_dim1], 
		lda, &b[*ilo + *ilo * b_dim1], ldb, &vl[vl_offset], ldvl, &vr[
		vr_offset], ldvr, &ierr);
    }

/*     Perform QZ algorithm (Compute eigenvalues, and optionally, the */
/*     Schur forms and Schur vectors) */
/*     (Workspace: need N) */

    if (ilv || ! wantsn) {
	*(unsigned char *)chtemp = 'S';
    } else {
	*(unsigned char *)chtemp = 'E';
    }

    hhgeqz_(chtemp, jobvl, jobvr, n, ilo, ihi, &a[a_offset], lda, &b[b_offset]
	    , ldb, &alphar[1], &alphai[1], &beta[1], &vl[vl_offset], ldvl, &
	    vr[vr_offset], ldvr, &work[1], lwork, &ierr);
    if (ierr != 0) {
	if (ierr > 0 && ierr <= *n) {
	    *info = ierr;
	} else if (ierr > *n && ierr <= *n << 1) {
	    *info = ierr - *n;
	} else {
	    *info = *n + 1;
	}
	goto L130;
    }

/*     Compute Eigenvectors and estimate condition numbers if desired */
/*     (Workspace: DTGEVC: need 6*N */
/*                 DTGSNA: need 2*N*(N+2)+16 if SENSE = 'V' or 'B', */
/*                         need N otherwise ) */

    if (ilv || ! wantsn) {
	if (ilv) {
	    if (ilvl) {
		if (ilvr) {
		    *(unsigned char *)chtemp = 'B';
		} else {
		    *(unsigned char *)chtemp = 'L';
		}
	    } else {
		*(unsigned char *)chtemp = 'R';
	    }

	    htgevc_(chtemp, "B", ldumma, n, &a[a_offset], lda, &b[b_offset], 
		    ldb, &vl[vl_offset], ldvl, &vr[vr_offset], ldvr, n, &in, &
		    work[1], &ierr);
	    if (ierr != 0) {
		*info = *n + 2;
		goto L130;
	    }
	}

	if (! wantsn) {

/*           compute eigenvectors (DTGEVC) and estimate condition */
/*           numbers (DTGSNA). Note that the definition of the condition */
/*           number is not invariant under transformation (u,v) to */
/*           (Q*u, Z*v), where (u,v) are eigenvectors of the generalized */
/*           Schur form (S,T), Q and Z are orthogonal matrices. In order */
/*           to avoid using extra 2*N*N workspace, we have to recalculate */
/*           eigenvectors and estimate one condition numbers at a time. */

	    pair = FALSE_;
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {

		if (pair) {
		    pair = FALSE_;
		    goto L20;
		}
		mm = 1;
		if (i__ < *n) {
		    if (a[i__ + 1 + i__ * a_dim1] != 0.) {
			pair = TRUE_;
			mm = 2;
		    }
		}

		i__2 = *n;
		for (j = 1; j <= i__2; ++j) {
		    bwork[j] = FALSE_;
/* L10: */
		}
		if (mm == 1) {
		    bwork[i__] = TRUE_;
		} else if (mm == 2) {
		    bwork[i__] = TRUE_;
		    bwork[i__ + 1] = TRUE_;
		}

		iwrk = mm * *n + 1;
		iwrk1 = iwrk + mm * *n;

/*              Compute a pair of left and right eigenvectors. */
/*              (compute workspace: need up to 4*N + 6*N) */

		if (wantse || wantsb) {
		    htgevc_("B", "S", &bwork[1], n, &a[a_offset], lda, &b[
			    b_offset], ldb, &work[1], n, &work[iwrk], n, &mm, 
			    &m, &work[iwrk1], &ierr);
		    if (ierr != 0) {
			*info = *n + 2;
			goto L130;
		    }
		}

		i__2 = *lwork - iwrk1 + 1;
		htgsna_(sense, "S", &bwork[1], n, &a[a_offset], lda, &b[
			b_offset], ldb, &work[1], n, &work[iwrk], n, &rconde[
			i__], &rcondv[i__], &mm, &m, &work[iwrk1], &i__2, &
			iwork[1], &ierr);

L20:
		;
	    }
	}
    }

/*     Undo balancing on VL and VR and normalization */
/*     (Workspace: none needed) */

    if (ilvl) {
	hggbak_(balanc, "L", n, ilo, ihi, &lscale[1], &rscale[1], n, &vl[
		vl_offset], ldvl, &ierr);

	i__1 = *n;
	for (jc = 1; jc <= i__1; ++jc) {
	    if (alphai[jc] < 0.) {
		goto L70;
	    }
	    temp = 0.;
	    if (alphai[jc] == 0.) {
		i__2 = *n;
		for (jr = 1; jr <= i__2; ++jr) {
/* Computing MAX */
		    d__2 = temp, d__3 = (d__1 = vl[jr + jc * vl_dim1], abs(
			    d__1));
		    temp = f2cmax(d__2,d__3);
/* L30: */
		}
	    } else {
		i__2 = *n;
		for (jr = 1; jr <= i__2; ++jr) {
/* Computing MAX */
		    d__3 = temp, d__4 = (d__1 = vl[jr + jc * vl_dim1], abs(
			    d__1)) + (d__2 = vl[jr + (jc + 1) * vl_dim1], abs(
			    d__2));
		    temp = f2cmax(d__3,d__4);
/* L40: */
		}
	    }
	    if (temp < smlnum) {
		goto L70;
	    }
	    temp = 1. / temp;
	    if (alphai[jc] == 0.) {
		i__2 = *n;
		for (jr = 1; jr <= i__2; ++jr) {
		    vl[jr + jc * vl_dim1] *= temp;
/* L50: */
		}
	    } else {
		i__2 = *n;
		for (jr = 1; jr <= i__2; ++jr) {
		    vl[jr + jc * vl_dim1] *= temp;
		    vl[jr + (jc + 1) * vl_dim1] *= temp;
/* L60: */
		}
	    }
L70:
	    ;
	}
    }
    if (ilvr) {
	hggbak_(balanc, "R", n, ilo, ihi, &lscale[1], &rscale[1], n, &vr[
		vr_offset], ldvr, &ierr);
	i__1 = *n;
	for (jc = 1; jc <= i__1; ++jc) {
	    if (alphai[jc] < 0.) {
		goto L120;
	    }
	    temp = 0.;
	    if (alphai[jc] == 0.) {
		i__2 = *n;
		for (jr = 1; jr <= i__2; ++jr) {
/* Computing MAX */
		    d__2 = temp, d__3 = (d__1 = vr[jr + jc * vr_dim1], abs(
			    d__1));
		    temp = f2cmax(d__2,d__3);
/* L80: */
		}
	    } else {
		i__2 = *n;
		for (jr = 1; jr <= i__2; ++jr) {
/* Computing MAX */
		    d__3 = temp, d__4 = (d__1 = vr[jr + jc * vr_dim1], abs(
			    d__1)) + (d__2 = vr[jr + (jc + 1) * vr_dim1], abs(
			    d__2));
		    temp = f2cmax(d__3,d__4);
/* L90: */
		}
	    }
	    if (temp < smlnum) {
		goto L120;
	    }
	    temp = 1. / temp;
	    if (alphai[jc] == 0.) {
		i__2 = *n;
		for (jr = 1; jr <= i__2; ++jr) {
		    vr[jr + jc * vr_dim1] *= temp;
/* L100: */
		}
	    } else {
		i__2 = *n;
		for (jr = 1; jr <= i__2; ++jr) {
		    vr[jr + jc * vr_dim1] *= temp;
		    vr[jr + (jc + 1) * vr_dim1] *= temp;
/* L110: */
		}
	    }
L120:
	    ;
	}
    }

/*     Undo scaling if necessary */

L130:

    if (ilascl) {
	hlascl_("G", &c__0, &c__0, &anrmto, &anrm, n, &c__1, &alphar[1], n, &
		ierr);
	hlascl_("G", &c__0, &c__0, &anrmto, &anrm, n, &c__1, &alphai[1], n, &
		ierr);
    }

    if (ilbscl) {
	hlascl_("G", &c__0, &c__0, &bnrmto, &bnrm, n, &c__1, &beta[1], n, &
		ierr);
    }

    work[1] = (halfreal) maxwrk;
    return;

/*     End of DGGEVX */

} /* hggevx_ */

