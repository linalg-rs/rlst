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
static halfcomplex c_b19 = {1.,0.};
static halfcomplex c_b20 = {0.,0.};
static logical c_false = FALSE_;
static integer c__3 = 3;

/* > \brief \b ZTGSNA */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZTGSNA + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ztgsna.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ztgsna.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ztgsna.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZTGSNA( JOB, HOWMNY, SELECT, N, A, LDA, B, LDB, VL, */
/*                          LDVL, VR, LDVR, S, DIF, MM, M, WORK, LWORK, */
/*                          IWORK, INFO ) */

/*       CHARACTER          HOWMNY, JOB */
/*       INTEGER            INFO, LDA, LDB, LDVL, LDVR, LWORK, M, MM, N */
/*       LOGICAL            SELECT( * ) */
/*       INTEGER            IWORK( * ) */
/*       DOUBLE PRECISION   DIF( * ), S( * ) */
/*       COMPLEX*16         A( LDA, * ), B( LDB, * ), VL( LDVL, * ), */
/*      $                   VR( LDVR, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZTGSNA estimates reciprocal condition numbers for specified */
/* > eigenvalues and/or eigenvectors of a matrix pair (A, B). */
/* > */
/* > (A, B) must be in generalized Schur canonical form, that is, A and */
/* > B are both upper triangular. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] JOB */
/* > \verbatim */
/* >          JOB is CHARACTER*1 */
/* >          Specifies whether condition numbers are required for */
/* >          eigenvalues (S) or eigenvectors (DIF): */
/* >          = 'E': for eigenvalues only (S); */
/* >          = 'V': for eigenvectors only (DIF); */
/* >          = 'B': for both eigenvalues and eigenvectors (S and DIF). */
/* > \endverbatim */
/* > */
/* > \param[in] HOWMNY */
/* > \verbatim */
/* >          HOWMNY is CHARACTER*1 */
/* >          = 'A': compute condition numbers for all eigenpairs; */
/* >          = 'S': compute condition numbers for selected eigenpairs */
/* >                 specified by the array SELECT. */
/* > \endverbatim */
/* > */
/* > \param[in] SELECT */
/* > \verbatim */
/* >          SELECT is LOGICAL array, dimension (N) */
/* >          If HOWMNY = 'S', SELECT specifies the eigenpairs for which */
/* >          condition numbers are required. To select condition numbers */
/* >          for the corresponding j-th eigenvalue and/or eigenvector, */
/* >          SELECT(j) must be set to .TRUE.. */
/* >          If HOWMNY = 'A', SELECT is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the square matrix pair (A, B). N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          The upper triangular matrix A in the pair (A,B). */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A. LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] B */
/* > \verbatim */
/* >          B is COMPLEX*16 array, dimension (LDB,N) */
/* >          The upper triangular matrix B in the pair (A, B). */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B. LDB >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] VL */
/* > \verbatim */
/* >          VL is COMPLEX*16 array, dimension (LDVL,M) */
/* >          IF JOB = 'E' or 'B', VL must contain left eigenvectors of */
/* >          (A, B), corresponding to the eigenpairs specified by HOWMNY */
/* >          and SELECT.  The eigenvectors must be stored in consecutive */
/* >          columns of VL, as returned by ZTGEVC. */
/* >          If JOB = 'V', VL is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVL */
/* > \verbatim */
/* >          LDVL is INTEGER */
/* >          The leading dimension of the array VL. LDVL >= 1; and */
/* >          If JOB = 'E' or 'B', LDVL >= N. */
/* > \endverbatim */
/* > */
/* > \param[in] VR */
/* > \verbatim */
/* >          VR is COMPLEX*16 array, dimension (LDVR,M) */
/* >          IF JOB = 'E' or 'B', VR must contain right eigenvectors of */
/* >          (A, B), corresponding to the eigenpairs specified by HOWMNY */
/* >          and SELECT.  The eigenvectors must be stored in consecutive */
/* >          columns of VR, as returned by ZTGEVC. */
/* >          If JOB = 'V', VR is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVR */
/* > \verbatim */
/* >          LDVR is INTEGER */
/* >          The leading dimension of the array VR. LDVR >= 1; */
/* >          If JOB = 'E' or 'B', LDVR >= N. */
/* > \endverbatim */
/* > */
/* > \param[out] S */
/* > \verbatim */
/* >          S is DOUBLE PRECISION array, dimension (MM) */
/* >          If JOB = 'E' or 'B', the reciprocal condition numbers of the */
/* >          selected eigenvalues, stored in consecutive elements of the */
/* >          array. */
/* >          If JOB = 'V', S is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[out] DIF */
/* > \verbatim */
/* >          DIF is DOUBLE PRECISION array, dimension (MM) */
/* >          If JOB = 'V' or 'B', the estimated reciprocal condition */
/* >          numbers of the selected eigenvectors, stored in consecutive */
/* >          elements of the array. */
/* >          If the eigenvalues cannot be reordered to compute DIF(j), */
/* >          DIF(j) is set to 0; this can only occur when the true value */
/* >          would be very small anyway. */
/* >          For each eigenvalue/vector specified by SELECT, DIF stores */
/* >          a Frobenius norm-based estimate of Difl. */
/* >          If JOB = 'E', DIF is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] MM */
/* > \verbatim */
/* >          MM is INTEGER */
/* >          The number of elements in the arrays S and DIF. MM >= M. */
/* > \endverbatim */
/* > */
/* > \param[out] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of elements of the arrays S and DIF used to store */
/* >          the specified condition numbers; for each selected eigenvalue */
/* >          one element is used. If HOWMNY = 'A', M is set to N. */
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
/* >          The dimension of the array WORK. LWORK >= f2cmax(1,N). */
/* >          If JOB = 'V' or 'B', LWORK >= f2cmax(1,2*N*N). */
/* > \endverbatim */
/* > */
/* > \param[out] IWORK */
/* > \verbatim */
/* >          IWORK is INTEGER array, dimension (N+2) */
/* >          If JOB = 'E', IWORK is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0: Successful exit */
/* >          < 0: If INFO = -i, the i-th argument had an illegal value */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16OTHERcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  The reciprocal of the condition number of the i-th generalized */
/* >  eigenvalue w = (a, b) is defined as */
/* > */
/* >          S(I) = (|v**HAu|**2 + |v**HBu|**2)**(1/2) / (norm(u)*norm(v)) */
/* > */
/* >  where u and v are the right and left eigenvectors of (A, B) */
/* >  corresponding to w; |z| denotes the absolute value of the complex */
/* >  number, and norm(u) denotes the 2-norm of the vector u. The pair */
/* >  (a, b) corresponds to an eigenvalue w = a/b (= v**HAu/v**HBu) of the */
/* >  matrix pair (A, B). If both a and b equal zero, then (A,B) is */
/* >  singular and S(I) = -1 is returned. */
/* > */
/* >  An approximate error bound on the chordal distance between the i-th */
/* >  computed generalized eigenvalue w and the corresponding exact */
/* >  eigenvalue lambda is */
/* > */
/* >          chord(w, lambda) <=   EPS * norm(A, B) / S(I), */
/* > */
/* >  where EPS is the machine precision. */
/* > */
/* >  The reciprocal of the condition number of the right eigenvector u */
/* >  and left eigenvector v corresponding to the generalized eigenvalue w */
/* >  is defined as follows. Suppose */
/* > */
/* >                   (A, B) = ( a   *  ) ( b  *  )  1 */
/* >                            ( 0  A22 ),( 0 B22 )  n-1 */
/* >                              1  n-1     1 n-1 */
/* > */
/* >  Then the reciprocal condition number DIF(I) is */
/* > */
/* >          Difl[(a, b), (A22, B22)]  = sigma-f2cmin( Zl ) */
/* > */
/* >  where sigma-f2cmin(Zl) denotes the smallest singular value of */
/* > */
/* >         Zl = [ kron(a, In-1) -kron(1, A22) ] */
/* >              [ kron(b, In-1) -kron(1, B22) ]. */
/* > */
/* >  Here In-1 is the identity matrix of size n-1 and X**H is the conjugate */
/* >  transpose of X. kron(X, Y) is the Kronecker product between the */
/* >  matrices X and Y. */
/* > */
/* >  We approximate the smallest singular value of Zl with an upper */
/* >  bound. This is done by ZLATDF. */
/* > */
/* >  An approximate error bound for a computed eigenvector VL(i) or */
/* >  VR(i) is given by */
/* > */
/* >                      EPS * norm(A, B) / DIF(i). */
/* > */
/* >  See ref. [2-3] for more details and further references. */
/* > \endverbatim */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >     Bo Kagstrom and Peter Poromaa, Department of Computing Science, */
/* >     Umea University, S-901 87 Umea, Sweden. */

/* > \par References: */
/*  ================ */
/* > */
/* > \verbatim */
/* > */
/* >  [1] B. Kagstrom; A Direct Method for Reordering Eigenvalues in the */
/* >      Generalized Real Schur Form of a Regular Matrix Pair (A, B), in */
/* >      M.S. Moonen et al (eds), Linear Algebra for Large Scale and */
/* >      Real-Time Applications, Kluwer Academic Publ. 1993, pp 195-218. */
/* > */
/* >  [2] B. Kagstrom and P. Poromaa; Computing Eigenspaces with Specified */
/* >      Eigenvalues of a Regular Matrix Pair (A, B) and Condition */
/* >      Estimation: Theory, Algorithms and Software, Report */
/* >      UMINF - 94.04, Department of Computing Science, Umea University, */
/* >      S-901 87 Umea, Sweden, 1994. Also as LAPACK Working Note 87. */
/* >      To appear in Numerical Algorithms, 1996. */
/* > */
/* >  [3] B. Kagstrom and P. Poromaa, LAPACK-Style Algorithms and Software */
/* >      for Solving the Generalized Sylvester Equation and Estimating the */
/* >      Separation between Regular Matrix Pairs, Report UMINF - 93.23, */
/* >      Department of Computing Science, Umea University, S-901 87 Umea, */
/* >      Sweden, December 1993, Revised April 1994, Also as LAPACK Working */
/* >      Note 75. */
/* >      To appear in ACM Trans. on Math. Software, Vol 22, No 1, 1996. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  ktgsna_(char *job, char *howmny, logical *select, 
	integer *n, halfcomplex *a, integer *lda, halfcomplex *b, integer 
	*ldb, halfcomplex *vl, integer *ldvl, halfcomplex *vr, integer *
	ldvr, halfreal *s, halfreal *dif, integer *mm, integer *m, 
	halfcomplex *work, integer *lwork, integer *iwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, vl_dim1, vl_offset, vr_dim1, 
	    vr_offset, i__1;
    halfreal d__1, d__2;
    halfcomplex z__1;

    /* Local variables */
    integer i__, k, n1, n2, ks;
    halfreal eps, cond;
    integer ierr, ifst;
    halfreal lnrm;
    halfcomplex yhax, yhbx;
    integer ilst;
    halfreal rnrm, scale;
    extern logical lsame_(char *, char *);
    extern /* Double Complex */ VOID whotc_(halfcomplex *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, integer *);
    integer lwmin;
    extern void  kgemv_(char *, integer *, integer *, 
	    halfcomplex *, halfcomplex *, integer *, halfcomplex *, 
	    integer *, halfcomplex *, halfcomplex *, integer *);
    logical wants;
    halfcomplex dummy[1];
    extern halfreal hlapy2_(halfreal *, halfreal *);
    extern void  hlabad_(halfreal *, halfreal *);
    halfcomplex dummy1[1];
    extern halfreal hwnrm2_(integer *, halfcomplex *, integer *), hlamch_(
	    char *);
    extern void  xerbla_(char *, integer *);
    halfreal bignum;
    logical wantbh, wantdf, somcon;
    extern void  klacpy_(char *, integer *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, integer *), 
	    ktgexc_(logical *, logical *, integer *, halfcomplex *, integer 
	    *, halfcomplex *, integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *, integer *, integer *, integer *);
    halfreal smlnum;
    logical lquery;
    extern void  ktgsyl_(char *, integer *, integer *, integer 
	    *, halfcomplex *, integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, integer *, 
	    halfreal *, halfreal *, halfcomplex *, integer *, integer *,
	     integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Decode and test the input parameters */

    /* Parameter adjustments */
    --select;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1;
    vr -= vr_offset;
    --s;
    --dif;
    --work;
    --iwork;

    /* Function Body */
    wantbh = lsame_(job, "B");
    wants = lsame_(job, "E") || wantbh;
    wantdf = lsame_(job, "V") || wantbh;

    somcon = lsame_(howmny, "S");

    *info = 0;
    lquery = *lwork == -1;

    if (! wants && ! wantdf) {
	*info = -1;
    } else if (! lsame_(howmny, "A") && ! somcon) {
	*info = -2;
    } else if (*n < 0) {
	*info = -4;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -6;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -8;
    } else if (wants && *ldvl < *n) {
	*info = -10;
    } else if (wants && *ldvr < *n) {
	*info = -12;
    } else {

/*        Set M to the number of eigenpairs for which condition numbers */
/*        are required, and test MM. */

	if (somcon) {
	    *m = 0;
	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {
		if (select[k]) {
		    ++(*m);
		}
/* L10: */
	    }
	} else {
	    *m = *n;
	}

	if (*n == 0) {
	    lwmin = 1;
	} else if (lsame_(job, "V") || lsame_(job, 
		"B")) {
	    lwmin = (*n << 1) * *n;
	} else {
	    lwmin = *n;
	}
	work[1].r = (halfreal) lwmin, work[1].i = 0.;

	if (*mm < *m) {
	    *info = -15;
	} else if (*lwork < lwmin && ! lquery) {
	    *info = -18;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZTGSNA", &i__1);
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
    smlnum = hlamch_("S") / eps;
    bignum = 1. / smlnum;
    hlabad_(&smlnum, &bignum);
    ks = 0;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {

/*        Determine whether condition numbers are required for the k-th */
/*        eigenpair. */

	if (somcon) {
	    if (! select[k]) {
		goto L20;
	    }
	}

	++ks;

	if (wants) {

/*           Compute the reciprocal condition number of the k-th */
/*           eigenvalue. */

	    rnrm = hwnrm2_(n, &vr[ks * vr_dim1 + 1], &c__1);
	    lnrm = hwnrm2_(n, &vl[ks * vl_dim1 + 1], &c__1);
	    kgemv_("N", n, n, &c_b19, &a[a_offset], lda, &vr[ks * vr_dim1 + 1]
		    , &c__1, &c_b20, &work[1], &c__1);
	    whotc_(&z__1, n, &work[1], &c__1, &vl[ks * vl_dim1 + 1], &c__1);
	    yhax.r = z__1.r, yhax.i = z__1.i;
	    kgemv_("N", n, n, &c_b19, &b[b_offset], ldb, &vr[ks * vr_dim1 + 1]
		    , &c__1, &c_b20, &work[1], &c__1);
	    whotc_(&z__1, n, &work[1], &c__1, &vl[ks * vl_dim1 + 1], &c__1);
	    yhbx.r = z__1.r, yhbx.i = z__1.i;
	    d__1 = z_abs(&yhax);
	    d__2 = z_abs(&yhbx);
	    cond = hlapy2_(&d__1, &d__2);
	    if (cond == 0.) {
		s[ks] = -1.;
	    } else {
		s[ks] = cond / (rnrm * lnrm);
	    }
	}

	if (wantdf) {
	    if (*n == 1) {
		d__1 = z_abs(&a[a_dim1 + 1]);
		d__2 = z_abs(&b[b_dim1 + 1]);
		dif[ks] = hlapy2_(&d__1, &d__2);
	    } else {

/*              Estimate the reciprocal condition number of the k-th */
/*              eigenvectors. */

/*              Copy the matrix (A, B) to the array WORK and move the */
/*              (k,k)th pair to the (1,1) position. */

		klacpy_("Full", n, n, &a[a_offset], lda, &work[1], n);
		klacpy_("Full", n, n, &b[b_offset], ldb, &work[*n * *n + 1], 
			n);
		ifst = k;
		ilst = 1;

		ktgexc_(&c_false, &c_false, n, &work[1], n, &work[*n * *n + 1]
			, n, dummy, &c__1, dummy1, &c__1, &ifst, &ilst, &ierr)
			;

		if (ierr > 0) {

/*                 Ill-conditioned problem - swap rejected. */

		    dif[ks] = 0.;
		} else {

/*                 Reordering successful, solve generalized Sylvester */
/*                 equation for R and L, */
/*                            A22 * R - L * A11 = A12 */
/*                            B22 * R - L * B11 = B12, */
/*                 and compute estimate of Difl[(A11,B11), (A22, B22)]. */

		    n1 = 1;
		    n2 = *n - n1;
		    i__ = *n * *n + 1;
		    ktgsyl_("N", &c__3, &n2, &n1, &work[*n * n1 + n1 + 1], n, 
			    &work[1], n, &work[n1 + 1], n, &work[*n * n1 + n1 
			    + i__], n, &work[i__], n, &work[n1 + i__], n, &
			    scale, &dif[ks], dummy, &c__1, &iwork[1], &ierr);
		}
	    }
	}

L20:
	;
    }
    work[1].r = (halfreal) lwmin, work[1].i = 0.;
    return;

/*     End of ZTGSNA */

} /* ktgsna_ */

