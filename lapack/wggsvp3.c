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

static quadcomplex c_b1 = {0.,0.};
static quadcomplex c_b2 = {1.,0.};
static integer c_n1 = -1;

/* > \brief \b ZGGSVP3 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZGGSVP3 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zggsvp3
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zggsvp3
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zggsvp3
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZGGSVP3( JOBU, JOBV, JOBQ, M, P, N, A, LDA, B, LDB, */
/*                           TOLA, TOLB, K, L, U, LDU, V, LDV, Q, LDQ, */
/*                           IWORK, RWORK, TAU, WORK, LWORK, INFO ) */

/*       CHARACTER          JOBQ, JOBU, JOBV */
/*       INTEGER            INFO, K, L, LDA, LDB, LDQ, LDU, LDV, M, N, P, LWORK */
/*       DOUBLE PRECISION   TOLA, TOLB */
/*       INTEGER            IWORK( * ) */
/*       DOUBLE PRECISION   RWORK( * ) */
/*       COMPLEX*16         A( LDA, * ), B( LDB, * ), Q( LDQ, * ), */
/*      $                   TAU( * ), U( LDU, * ), V( LDV, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZGGSVP3 computes unitary matrices U, V and Q such that */
/* > */
/* >                    N-K-L  K    L */
/* >  U**H*A*Q =     K ( 0    A12  A13 )  if M-K-L >= 0; */
/* >                 L ( 0     0   A23 ) */
/* >             M-K-L ( 0     0    0  ) */
/* > */
/* >                  N-K-L  K    L */
/* >         =     K ( 0    A12  A13 )  if M-K-L < 0; */
/* >             M-K ( 0     0   A23 ) */
/* > */
/* >                  N-K-L  K    L */
/* >  V**H*B*Q =   L ( 0     0   B13 ) */
/* >             P-L ( 0     0    0  ) */
/* > */
/* > where the K-by-K matrix A12 and L-by-L matrix B13 are nonsingular */
/* > upper triangular; A23 is L-by-L upper triangular if M-K-L >= 0, */
/* > otherwise A23 is (M-K)-by-L upper trapezoidal.  K+L = the effective */
/* > numerical rank of the (M+P)-by-N matrix (A**H,B**H)**H. */
/* > */
/* > This decomposition is the preprocessing step for computing the */
/* > Generalized Singular Value Decomposition (GSVD), see subroutine */
/* > ZGGSVD3. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] JOBU */
/* > \verbatim */
/* >          JOBU is CHARACTER*1 */
/* >          = 'U':  Unitary matrix U is computed; */
/* >          = 'N':  U is not computed. */
/* > \endverbatim */
/* > */
/* > \param[in] JOBV */
/* > \verbatim */
/* >          JOBV is CHARACTER*1 */
/* >          = 'V':  Unitary matrix V is computed; */
/* >          = 'N':  V is not computed. */
/* > \endverbatim */
/* > */
/* > \param[in] JOBQ */
/* > \verbatim */
/* >          JOBQ is CHARACTER*1 */
/* >          = 'Q':  Unitary matrix Q is computed; */
/* >          = 'N':  Q is not computed. */
/* > \endverbatim */
/* > */
/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix A.  M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] P */
/* > \verbatim */
/* >          P is INTEGER */
/* >          The number of rows of the matrix B.  P >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrices A and B.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          On entry, the M-by-N matrix A. */
/* >          On exit, A contains the triangular (or trapezoidal) matrix */
/* >          described in the Purpose section. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A. LDA >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is COMPLEX*16 array, dimension (LDB,N) */
/* >          On entry, the P-by-N matrix B. */
/* >          On exit, B contains the triangular matrix described in */
/* >          the Purpose section. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B. LDB >= f2cmax(1,P). */
/* > \endverbatim */
/* > */
/* > \param[in] TOLA */
/* > \verbatim */
/* >          TOLA is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[in] TOLB */
/* > \verbatim */
/* >          TOLB is DOUBLE PRECISION */
/* > */
/* >          TOLA and TOLB are the thresholds to determine the effective */
/* >          numerical rank of matrix B and a subblock of A. Generally, */
/* >          they are set to */
/* >             TOLA = MAX(M,N)*norm(A)*MAZHEPS, */
/* >             TOLB = MAX(P,N)*norm(B)*MAZHEPS. */
/* >          The size of TOLA and TOLB may affect the size of backward */
/* >          errors of the decomposition. */
/* > \endverbatim */
/* > */
/* > \param[out] K */
/* > \verbatim */
/* >          K is INTEGER */
/* > \endverbatim */
/* > */
/* > \param[out] L */
/* > \verbatim */
/* >          L is INTEGER */
/* > */
/* >          On exit, K and L specify the dimension of the subblocks */
/* >          described in Purpose section. */
/* >          K + L = effective numerical rank of (A**H,B**H)**H. */
/* > \endverbatim */
/* > */
/* > \param[out] U */
/* > \verbatim */
/* >          U is COMPLEX*16 array, dimension (LDU,M) */
/* >          If JOBU = 'U', U contains the unitary matrix U. */
/* >          If JOBU = 'N', U is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDU */
/* > \verbatim */
/* >          LDU is INTEGER */
/* >          The leading dimension of the array U. LDU >= f2cmax(1,M) if */
/* >          JOBU = 'U'; LDU >= 1 otherwise. */
/* > \endverbatim */
/* > */
/* > \param[out] V */
/* > \verbatim */
/* >          V is COMPLEX*16 array, dimension (LDV,P) */
/* >          If JOBV = 'V', V contains the unitary matrix V. */
/* >          If JOBV = 'N', V is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDV */
/* > \verbatim */
/* >          LDV is INTEGER */
/* >          The leading dimension of the array V. LDV >= f2cmax(1,P) if */
/* >          JOBV = 'V'; LDV >= 1 otherwise. */
/* > \endverbatim */
/* > */
/* > \param[out] Q */
/* > \verbatim */
/* >          Q is COMPLEX*16 array, dimension (LDQ,N) */
/* >          If JOBQ = 'Q', Q contains the unitary matrix Q. */
/* >          If JOBQ = 'N', Q is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDQ */
/* > \verbatim */
/* >          LDQ is INTEGER */
/* >          The leading dimension of the array Q. LDQ >= f2cmax(1,N) if */
/* >          JOBQ = 'Q'; LDQ >= 1 otherwise. */
/* > \endverbatim */
/* > */
/* > \param[out] IWORK */
/* > \verbatim */
/* >          IWORK is INTEGER array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is DOUBLE PRECISION array, dimension (2*N) */
/* > \endverbatim */
/* > */
/* > \param[out] TAU */
/* > \verbatim */
/* >          TAU is COMPLEX*16 array, dimension (N) */
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
/* >          The dimension of the array WORK. */
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
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date August 2015 */

/* > \ingroup complex16OTHERcomputational */

/* > \par Further Details: */
/*  ===================== */

/* > \verbatim */
/* > */
/* >  The subroutine uses LAPACK subroutine ZGEQP3 for the QR factorization */
/* >  with column pivoting to detect the effective numerical rank of the */
/* >  a matrix. It may be replaced by a better rank determination strategy. */
/* > */
/* >  ZGGSVP3 replaces the deprecated subroutine ZGGSVP. */
/* > */
/* > \endverbatim */
/* > */
/* ===================================================================== */
void  wggsvp3_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, quadcomplex *a, integer *lda, quadcomplex 
	*b, integer *ldb, quadreal *tola, quadreal *tolb, integer *k, 
	integer *l, quadcomplex *u, integer *ldu, quadcomplex *v, integer 
	*ldv, quadcomplex *q, integer *ldq, integer *iwork, quadreal *
	rwork, quadcomplex *tau, quadcomplex *work, integer *lwork, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, q_dim1, q_offset, u_dim1, 
	    u_offset, v_dim1, v_offset, i__1, i__2, i__3;
    quadcomplex z__1;

    /* Local variables */
    integer i__, j;
    extern logical lsame_(char *, char *);
    logical wantq, wantu, wantv;
    extern void  wgeqp3_(integer *, integer *, quadcomplex *,
	     integer *, integer *, quadcomplex *, quadcomplex *, integer *
	    , quadreal *, integer *), wgeqr2_(integer *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, quadcomplex *, 
	    integer *), wgerq2_(integer *, integer *, quadcomplex *, 
	    integer *, quadcomplex *, quadcomplex *, integer *), wung2r_(
	    integer *, integer *, integer *, quadcomplex *, integer *, 
	    quadcomplex *, quadcomplex *, integer *), wunm2r_(char *, 
	    char *, integer *, integer *, integer *, quadcomplex *, integer 
	    *, quadcomplex *, quadcomplex *, integer *, quadcomplex *, 
	    integer *), wunmr2_(char *, char *, integer *, 
	    integer *, integer *, quadcomplex *, integer *, quadcomplex *,
	     quadcomplex *, integer *, quadcomplex *, integer *), xerbla_(char *, integer *), wlacpy_(char *, 
	    integer *, integer *, quadcomplex *, integer *, quadcomplex *,
	     integer *);
    logical forwrd;
    extern void  wlaset_(char *, integer *, integer *, 
	    quadcomplex *, quadcomplex *, quadcomplex *, integer *), wlapmt_(logical *, integer *, integer *, quadcomplex *,
	     integer *, integer *);
    integer lwkopt;
    logical lquery;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     August 2015 */



/*  ===================================================================== */


/*     Test the input parameters */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --iwork;
    --rwork;
    --tau;
    --work;

    /* Function Body */
    wantu = lsame_(jobu, "U");
    wantv = lsame_(jobv, "V");
    wantq = lsame_(jobq, "Q");
    forwrd = TRUE_;
    lquery = *lwork == -1;
    lwkopt = 1;

/*     Test the input arguments */

    *info = 0;
    if (! (wantu || lsame_(jobu, "N"))) {
	*info = -1;
    } else if (! (wantv || lsame_(jobv, "N"))) {
	*info = -2;
    } else if (! (wantq || lsame_(jobq, "N"))) {
	*info = -3;
    } else if (*m < 0) {
	*info = -4;
    } else if (*p < 0) {
	*info = -5;
    } else if (*n < 0) {
	*info = -6;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -8;
    } else if (*ldb < f2cmax(1,*p)) {
	*info = -10;
    } else if (*ldu < 1 || wantu && *ldu < *m) {
	*info = -16;
    } else if (*ldv < 1 || wantv && *ldv < *p) {
	*info = -18;
    } else if (*ldq < 1 || wantq && *ldq < *n) {
	*info = -20;
    } else if (*lwork < 1 && ! lquery) {
	*info = -24;
    }

/*     Compute workspace */

    if (*info == 0) {
	wgeqp3_(p, n, &b[b_offset], ldb, &iwork[1], &tau[1], &work[1], &c_n1, 
		&rwork[1], info);
	lwkopt = (integer) work[1].r;
	if (wantv) {
	    lwkopt = f2cmax(lwkopt,*p);
	}
/* Computing MAX */
	i__1 = lwkopt, i__2 = f2cmin(*n,*p);
	lwkopt = f2cmax(i__1,i__2);
	lwkopt = f2cmax(lwkopt,*m);
	if (wantq) {
	    lwkopt = f2cmax(lwkopt,*n);
	}
	wgeqp3_(m, n, &a[a_offset], lda, &iwork[1], &tau[1], &work[1], &c_n1, 
		&rwork[1], info);
/* Computing MAX */
	i__1 = lwkopt, i__2 = (integer) work[1].r;
	lwkopt = f2cmax(i__1,i__2);
	lwkopt = f2cmax(1,lwkopt);
	z__1.r = (quadreal) lwkopt, z__1.i = 0.;
	work[1].r = z__1.r, work[1].i = z__1.i;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGGSVP3", &i__1);
	return;
    }
    if (lquery) {
	return;
    }

/*     QR with column pivoting of B: B*P = V*( S11 S12 ) */
/*                                           (  0   0  ) */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	iwork[i__] = 0;
/* L10: */
    }
    wgeqp3_(p, n, &b[b_offset], ldb, &iwork[1], &tau[1], &work[1], lwork, &
	    rwork[1], info);

/*     Update A := A*P */

    wlapmt_(&forwrd, m, n, &a[a_offset], lda, &iwork[1]);

/*     Determine the effective rank of matrix B. */

    *l = 0;
    i__1 = f2cmin(*p,*n);
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (z_abs(&b[i__ + i__ * b_dim1]) > *tolb) {
	    ++(*l);
	}
/* L20: */
    }

    if (wantv) {

/*        Copy the details of V, and form V. */

	wlaset_("Full", p, p, &c_b1, &c_b1, &v[v_offset], ldv);
	if (*p > 1) {
	    i__1 = *p - 1;
	    wlacpy_("Lower", &i__1, n, &b[b_dim1 + 2], ldb, &v[v_dim1 + 2], 
		    ldv);
	}
	i__1 = f2cmin(*p,*n);
	wung2r_(p, p, &i__1, &v[v_offset], ldv, &tau[1], &work[1], info);
    }

/*     Clean up B */

    i__1 = *l - 1;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *l;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * b_dim1;
	    b[i__3].r = 0., b[i__3].i = 0.;
/* L30: */
	}
/* L40: */
    }
    if (*p > *l) {
	i__1 = *p - *l;
	wlaset_("Full", &i__1, n, &c_b1, &c_b1, &b[*l + 1 + b_dim1], ldb);
    }

    if (wantq) {

/*        Set Q = I and Update Q := Q*P */

	wlaset_("Full", n, n, &c_b1, &c_b2, &q[q_offset], ldq);
	wlapmt_(&forwrd, n, n, &q[q_offset], ldq, &iwork[1]);
    }

    if (*p >= *l && *n != *l) {

/*        RQ factorization of ( S11 S12 ) = ( 0 S12 )*Z */

	wgerq2_(l, n, &b[b_offset], ldb, &tau[1], &work[1], info);

/*        Update A := A*Z**H */

	wunmr2_("Right", "Conjugate transpose", m, n, l, &b[b_offset], ldb, &
		tau[1], &a[a_offset], lda, &work[1], info);
	if (wantq) {

/*           Update Q := Q*Z**H */

	    wunmr2_("Right", "Conjugate transpose", n, n, l, &b[b_offset], 
		    ldb, &tau[1], &q[q_offset], ldq, &work[1], info);
	}

/*        Clean up B */

	i__1 = *n - *l;
	wlaset_("Full", l, &i__1, &c_b1, &c_b1, &b[b_offset], ldb);
	i__1 = *n;
	for (j = *n - *l + 1; j <= i__1; ++j) {
	    i__2 = *l;
	    for (i__ = j - *n + *l + 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * b_dim1;
		b[i__3].r = 0., b[i__3].i = 0.;
/* L50: */
	    }
/* L60: */
	}

    }

/*     Let              N-L     L */
/*                A = ( A11    A12 ) M, */

/*     then the following does the complete QR decomposition of A11: */

/*              A11 = U*(  0  T12 )*P1**H */
/*                      (  0   0  ) */

    i__1 = *n - *l;
    for (i__ = 1; i__ <= i__1; ++i__) {
	iwork[i__] = 0;
/* L70: */
    }
    i__1 = *n - *l;
    wgeqp3_(m, &i__1, &a[a_offset], lda, &iwork[1], &tau[1], &work[1], lwork, 
	    &rwork[1], info);

/*     Determine the effective rank of A11 */

    *k = 0;
/* Computing MIN */
    i__2 = *m, i__3 = *n - *l;
    i__1 = f2cmin(i__2,i__3);
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (z_abs(&a[i__ + i__ * a_dim1]) > *tola) {
	    ++(*k);
	}
/* L80: */
    }

/*     Update A12 := U**H*A12, where A12 = A( 1:M, N-L+1:N ) */

/* Computing MIN */
    i__2 = *m, i__3 = *n - *l;
    i__1 = f2cmin(i__2,i__3);
    wunm2r_("Left", "Conjugate transpose", m, l, &i__1, &a[a_offset], lda, &
	    tau[1], &a[(*n - *l + 1) * a_dim1 + 1], lda, &work[1], info);

    if (wantu) {

/*        Copy the details of U, and form U */

	wlaset_("Full", m, m, &c_b1, &c_b1, &u[u_offset], ldu);
	if (*m > 1) {
	    i__1 = *m - 1;
	    i__2 = *n - *l;
	    wlacpy_("Lower", &i__1, &i__2, &a[a_dim1 + 2], lda, &u[u_dim1 + 2]
		    , ldu);
	}
/* Computing MIN */
	i__2 = *m, i__3 = *n - *l;
	i__1 = f2cmin(i__2,i__3);
	wung2r_(m, m, &i__1, &u[u_offset], ldu, &tau[1], &work[1], info);
    }

    if (wantq) {

/*        Update Q( 1:N, 1:N-L )  = Q( 1:N, 1:N-L )*P1 */

	i__1 = *n - *l;
	wlapmt_(&forwrd, n, &i__1, &q[q_offset], ldq, &iwork[1]);
    }

/*     Clean up A: set the strictly lower triangular part of */
/*     A(1:K, 1:K) = 0, and A( K+1:M, 1:N-L ) = 0. */

    i__1 = *k - 1;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *k;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    a[i__3].r = 0., a[i__3].i = 0.;
/* L90: */
	}
/* L100: */
    }
    if (*m > *k) {
	i__1 = *m - *k;
	i__2 = *n - *l;
	wlaset_("Full", &i__1, &i__2, &c_b1, &c_b1, &a[*k + 1 + a_dim1], lda);
    }

    if (*n - *l > *k) {

/*        RQ factorization of ( T11 T12 ) = ( 0 T12 )*Z1 */

	i__1 = *n - *l;
	wgerq2_(k, &i__1, &a[a_offset], lda, &tau[1], &work[1], info);

	if (wantq) {

/*           Update Q( 1:N,1:N-L ) = Q( 1:N,1:N-L )*Z1**H */

	    i__1 = *n - *l;
	    wunmr2_("Right", "Conjugate transpose", n, &i__1, k, &a[a_offset],
		     lda, &tau[1], &q[q_offset], ldq, &work[1], info);
	}

/*        Clean up A */

	i__1 = *n - *l - *k;
	wlaset_("Full", k, &i__1, &c_b1, &c_b1, &a[a_offset], lda);
	i__1 = *n - *l;
	for (j = *n - *l - *k + 1; j <= i__1; ++j) {
	    i__2 = *k;
	    for (i__ = j - *n + *l + *k + 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * a_dim1;
		a[i__3].r = 0., a[i__3].i = 0.;
/* L110: */
	    }
/* L120: */
	}

    }

    if (*m > *k) {

/*        QR factorization of A( K+1:M,N-L+1:N ) */

	i__1 = *m - *k;
	wgeqr2_(&i__1, l, &a[*k + 1 + (*n - *l + 1) * a_dim1], lda, &tau[1], &
		work[1], info);

	if (wantu) {

/*           Update U(:,K+1:M) := U(:,K+1:M)*U1 */

	    i__1 = *m - *k;
/* Computing MIN */
	    i__3 = *m - *k;
	    i__2 = f2cmin(i__3,*l);
	    wunm2r_("Right", "No transpose", m, &i__1, &i__2, &a[*k + 1 + (*n 
		    - *l + 1) * a_dim1], lda, &tau[1], &u[(*k + 1) * u_dim1 + 
		    1], ldu, &work[1], info);
	}

/*        Clean up */

	i__1 = *n;
	for (j = *n - *l + 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = j - *n + *k + *l + 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * a_dim1;
		a[i__3].r = 0., a[i__3].i = 0.;
/* L130: */
	    }
/* L140: */
	}

    }

    z__1.r = (quadreal) lwkopt, z__1.i = 0.;
    work[1].r = z__1.r, work[1].i = z__1.i;
    return;

/*     End of ZGGSVP3 */

} /* wggsvp3_ */

