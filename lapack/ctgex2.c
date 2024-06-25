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

static integer c__2 = 2;
static integer c__1 = 1;

/* > \brief \b CTGEX2 swaps adjacent diagonal blocks in an upper (quasi) triangular matrix pair by an unitary 
equivalence transformation. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CTGEX2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ctgex2.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ctgex2.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ctgex2.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CTGEX2( WANTQ, WANTZ, N, A, LDA, B, LDB, Q, LDQ, Z, */
/*                          LDZ, J1, INFO ) */

/*       LOGICAL            WANTQ, WANTZ */
/*       INTEGER            INFO, J1, LDA, LDB, LDQ, LDZ, N */
/*       COMPLEX            A( LDA, * ), B( LDB, * ), Q( LDQ, * ), */
/*      $                   Z( LDZ, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CTGEX2 swaps adjacent diagonal 1 by 1 blocks (A11,B11) and (A22,B22) */
/* > in an upper triangular matrix pair (A, B) by an unitary equivalence */
/* > transformation. */
/* > */
/* > (A, B) must be in generalized Schur canonical form, that is, A and */
/* > B are both upper triangular. */
/* > */
/* > Optionally, the matrices Q and Z of generalized Schur vectors are */
/* > updated. */
/* > */
/* >        Q(in) * A(in) * Z(in)**H = Q(out) * A(out) * Z(out)**H */
/* >        Q(in) * B(in) * Z(in)**H = Q(out) * B(out) * Z(out)**H */
/* > */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] WANTQ */
/* > \verbatim */
/* >          WANTQ is LOGICAL */
/* >          .TRUE. : update the left transformation matrix Q; */
/* >          .FALSE.: do not update Q. */
/* > \endverbatim */
/* > */
/* > \param[in] WANTZ */
/* > \verbatim */
/* >          WANTZ is LOGICAL */
/* >          .TRUE. : update the right transformation matrix Z; */
/* >          .FALSE.: do not update Z. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrices A and B. N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX array, dimension (LDA,N) */
/* >          On entry, the matrix A in the pair (A, B). */
/* >          On exit, the updated matrix A. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A. LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is COMPLEX array, dimension (LDB,N) */
/* >          On entry, the matrix B in the pair (A, B). */
/* >          On exit, the updated matrix B. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B. LDB >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in,out] Q */
/* > \verbatim */
/* >          Q is COMPLEX array, dimension (LDQ,N) */
/* >          If WANTQ = .TRUE, on entry, the unitary matrix Q. On exit, */
/* >          the updated matrix Q. */
/* >          Not referenced if WANTQ = .FALSE.. */
/* > \endverbatim */
/* > */
/* > \param[in] LDQ */
/* > \verbatim */
/* >          LDQ is INTEGER */
/* >          The leading dimension of the array Q. LDQ >= 1; */
/* >          If WANTQ = .TRUE., LDQ >= N. */
/* > \endverbatim */
/* > */
/* > \param[in,out] Z */
/* > \verbatim */
/* >          Z is COMPLEX array, dimension (LDZ,N) */
/* >          If WANTZ = .TRUE, on entry, the unitary matrix Z. On exit, */
/* >          the updated matrix Z. */
/* >          Not referenced if WANTZ = .FALSE.. */
/* > \endverbatim */
/* > */
/* > \param[in] LDZ */
/* > \verbatim */
/* >          LDZ is INTEGER */
/* >          The leading dimension of the array Z. LDZ >= 1; */
/* >          If WANTZ = .TRUE., LDZ >= N. */
/* > \endverbatim */
/* > */
/* > \param[in] J1 */
/* > \verbatim */
/* >          J1 is INTEGER */
/* >          The index to the first block (A11, B11). */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >           =0:  Successful exit. */
/* >           =1:  The transformed matrix pair (A, B) would be too far */
/* >                from generalized Schur form; the problem is ill- */
/* >                conditioned. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2017 */

/* > \ingroup complexGEauxiliary */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* >  In the current code both weak and strong stability tests are */
/* >  performed. The user can omit the strong stability test by changing */
/* >  the internal logical parameter WANDS to .FALSE.. See ref. [2] for */
/* >  details. */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >     Bo Kagstrom and Peter Poromaa, Department of Computing Science, */
/* >     Umea University, S-901 87 Umea, Sweden. */

/* > \par References: */
/*  ================ */
/* > */
/* >  [1] B. Kagstrom; A Direct Method for Reordering Eigenvalues in the */
/* >      Generalized Real Schur Form of a Regular Matrix Pair (A, B), in */
/* >      M.S. Moonen et al (eds), Linear Algebra for Large Scale and */
/* >      Real-Time Applications, Kluwer Academic Publ. 1993, pp 195-218. */
/* > \n */
/* >  [2] B. Kagstrom and P. Poromaa; Computing Eigenspaces with Specified */
/* >      Eigenvalues of a Regular Matrix Pair (A, B) and Condition */
/* >      Estimation: Theory, Algorithms and Software, Report UMINF-94.04, */
/* >      Department of Computing Science, Umea University, S-901 87 Umea, */
/* >      Sweden, 1994. Also as LAPACK Working Note 87. To appear in */
/* >      Numerical Algorithms, 1996. */
/* > */
/*  ===================================================================== */
void  ctgex2_(logical *wantq, logical *wantz, integer *n, 
	complex *a, integer *lda, complex *b, integer *ldb, complex *q, 
	integer *ldq, complex *z__, integer *ldz, integer *j1, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, q_dim1, q_offset, z_dim1, 
	    z_offset, i__1, i__2, i__3;
    real r__1;
    complex q__1, q__2, q__3;

    /* Local variables */
    complex f, g;
    integer i__, m;
    complex s[4]	/* was [2][2] */, t[4]	/* was [2][2] */;
    real cq, sa, sb, cz;
    complex sq;
    real ss, ws;
    complex sz;
    real eps, sum;
    logical weak;
    complex cdum;
    extern void  crot_(integer *, complex *, integer *, 
	    complex *, integer *, real *, complex *);
    complex work[8];
    real scale;
    extern real slamch_(char *);
    extern void  clacpy_(char *, integer *, integer *, complex 
	    *, integer *, complex *, integer *), clartg_(complex *, 
	    complex *, real *, complex *, complex *), classq_(integer *, 
	    complex *, integer *, real *, real *);
    real thresh, smlnum;
    logical strong;


/*  -- LAPACK auxiliary routine (version 3.7.1) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2017 */


/*  ===================================================================== */


    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;

    /* Function Body */
    *info = 0;

/*     Quick return if possible */

    if (*n <= 1) {
	return;
    }

    m = 2;
    weak = FALSE_;
    strong = FALSE_;

/*     Make a local copy of selected block in (A, B) */

    clacpy_("Full", &m, &m, &a[*j1 + *j1 * a_dim1], lda, s, &c__2);
    clacpy_("Full", &m, &m, &b[*j1 + *j1 * b_dim1], ldb, t, &c__2);

/*     Compute the threshold for testing the acceptance of swapping. */

    eps = slamch_("P");
    smlnum = slamch_("S") / eps;
    scale = 0.f;
    sum = 1.f;
    clacpy_("Full", &m, &m, s, &c__2, work, &m);
    clacpy_("Full", &m, &m, t, &c__2, &work[m * m], &m);
    i__1 = (m << 1) * m;
    classq_(&i__1, work, &c__1, &scale, &sum);
    sa = scale * M(sqrt)(sum);

/*     THRES has been changed from */
/*        THRESH = MAX( TEN*EPS*SA, SMLNUM ) */
/*     to */
/*        THRESH = MAX( TWENTY*EPS*SA, SMLNUM ) */
/*     on 04/01/10. */
/*     "Bug" reported by Ondra Kamenik, confirmed by Julie Langou, fixed by */
/*     Jim Demmel and Guillaume Revy. See forum post 1783. */

/* Computing MAX */
    r__1 = eps * 20.f * sa;
    thresh = f2cmax(r__1,smlnum);

/*     Compute unitary QL and RQ that swap 1-by-1 and 1-by-1 blocks */
/*     using Givens rotations and perform the swap tentatively. */

    q__2.r = s[3].r * t[0].r - s[3].i * t[0].i, q__2.i = s[3].r * t[0].i + s[
	    3].i * t[0].r;
    q__3.r = t[3].r * s[0].r - t[3].i * s[0].i, q__3.i = t[3].r * s[0].i + t[
	    3].i * s[0].r;
    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
    f.r = q__1.r, f.i = q__1.i;
    q__2.r = s[3].r * t[2].r - s[3].i * t[2].i, q__2.i = s[3].r * t[2].i + s[
	    3].i * t[2].r;
    q__3.r = t[3].r * s[2].r - t[3].i * s[2].i, q__3.i = t[3].r * s[2].i + t[
	    3].i * s[2].r;
    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
    g.r = q__1.r, g.i = q__1.i;
    sa = c_abs(&s[3]);
    sb = c_abs(&t[3]);
    clartg_(&g, &f, &cz, &sz, &cdum);
    q__1.r = -sz.r, q__1.i = -sz.i;
    sz.r = q__1.r, sz.i = q__1.i;
    r_cnjg(&q__1, &sz);
    crot_(&c__2, s, &c__1, &s[2], &c__1, &cz, &q__1);
    r_cnjg(&q__1, &sz);
    crot_(&c__2, t, &c__1, &t[2], &c__1, &cz, &q__1);
    if (sa >= sb) {
	clartg_(s, &s[1], &cq, &sq, &cdum);
    } else {
	clartg_(t, &t[1], &cq, &sq, &cdum);
    }
    crot_(&c__2, s, &c__2, &s[1], &c__2, &cq, &sq);
    crot_(&c__2, t, &c__2, &t[1], &c__2, &cq, &sq);

/*     Weak stability test: |S21| + |T21| <= O(EPS F-norm((S, T))) */

    ws = c_abs(&s[1]) + c_abs(&t[1]);
    weak = ws <= thresh;
    if (! weak) {
	goto L20;
    }

    if (TRUE_) {

/*        Strong stability test: */
/*           F-norm((A-QL**H*S*QR, B-QL**H*T*QR)) <= O(EPS*F-norm((A, B))) */

	clacpy_("Full", &m, &m, s, &c__2, work, &m);
	clacpy_("Full", &m, &m, t, &c__2, &work[m * m], &m);
	r_cnjg(&q__2, &sz);
	q__1.r = -q__2.r, q__1.i = -q__2.i;
	crot_(&c__2, work, &c__1, &work[2], &c__1, &cz, &q__1);
	r_cnjg(&q__2, &sz);
	q__1.r = -q__2.r, q__1.i = -q__2.i;
	crot_(&c__2, &work[4], &c__1, &work[6], &c__1, &cz, &q__1);
	q__1.r = -sq.r, q__1.i = -sq.i;
	crot_(&c__2, work, &c__2, &work[1], &c__2, &cq, &q__1);
	q__1.r = -sq.r, q__1.i = -sq.i;
	crot_(&c__2, &work[4], &c__2, &work[5], &c__2, &cq, &q__1);
	for (i__ = 1; i__ <= 2; ++i__) {
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    i__3 = *j1 + i__ - 1 + *j1 * a_dim1;
	    q__1.r = work[i__2].r - a[i__3].r, q__1.i = work[i__2].i - a[i__3]
		    .i;
	    work[i__1].r = q__1.r, work[i__1].i = q__1.i;
	    i__1 = i__ + 1;
	    i__2 = i__ + 1;
	    i__3 = *j1 + i__ - 1 + (*j1 + 1) * a_dim1;
	    q__1.r = work[i__2].r - a[i__3].r, q__1.i = work[i__2].i - a[i__3]
		    .i;
	    work[i__1].r = q__1.r, work[i__1].i = q__1.i;
	    i__1 = i__ + 3;
	    i__2 = i__ + 3;
	    i__3 = *j1 + i__ - 1 + *j1 * b_dim1;
	    q__1.r = work[i__2].r - b[i__3].r, q__1.i = work[i__2].i - b[i__3]
		    .i;
	    work[i__1].r = q__1.r, work[i__1].i = q__1.i;
	    i__1 = i__ + 5;
	    i__2 = i__ + 5;
	    i__3 = *j1 + i__ - 1 + (*j1 + 1) * b_dim1;
	    q__1.r = work[i__2].r - b[i__3].r, q__1.i = work[i__2].i - b[i__3]
		    .i;
	    work[i__1].r = q__1.r, work[i__1].i = q__1.i;
/* L10: */
	}
	scale = 0.f;
	sum = 1.f;
	i__1 = (m << 1) * m;
	classq_(&i__1, work, &c__1, &scale, &sum);
	ss = scale * M(sqrt)(sum);
	strong = ss <= thresh;
	if (! strong) {
	    goto L20;
	}
    }

/*     If the swap is accepted ("weakly" and "strongly"), apply the */
/*     equivalence transformations to the original matrix pair (A,B) */

    i__1 = *j1 + 1;
    r_cnjg(&q__1, &sz);
    crot_(&i__1, &a[*j1 * a_dim1 + 1], &c__1, &a[(*j1 + 1) * a_dim1 + 1], &
	    c__1, &cz, &q__1);
    i__1 = *j1 + 1;
    r_cnjg(&q__1, &sz);
    crot_(&i__1, &b[*j1 * b_dim1 + 1], &c__1, &b[(*j1 + 1) * b_dim1 + 1], &
	    c__1, &cz, &q__1);
    i__1 = *n - *j1 + 1;
    crot_(&i__1, &a[*j1 + *j1 * a_dim1], lda, &a[*j1 + 1 + *j1 * a_dim1], lda,
	     &cq, &sq);
    i__1 = *n - *j1 + 1;
    crot_(&i__1, &b[*j1 + *j1 * b_dim1], ldb, &b[*j1 + 1 + *j1 * b_dim1], ldb,
	     &cq, &sq);

/*     Set  N1 by N2 (2,1) blocks to 0 */

    i__1 = *j1 + 1 + *j1 * a_dim1;
    a[i__1].r = 0.f, a[i__1].i = 0.f;
    i__1 = *j1 + 1 + *j1 * b_dim1;
    b[i__1].r = 0.f, b[i__1].i = 0.f;

/*     Accumulate transformations into Q and Z if requested. */

    if (*wantz) {
	r_cnjg(&q__1, &sz);
	crot_(n, &z__[*j1 * z_dim1 + 1], &c__1, &z__[(*j1 + 1) * z_dim1 + 1], 
		&c__1, &cz, &q__1);
    }
    if (*wantq) {
	r_cnjg(&q__1, &sq);
	crot_(n, &q[*j1 * q_dim1 + 1], &c__1, &q[(*j1 + 1) * q_dim1 + 1], &
		c__1, &cq, &q__1);
    }

/*     Exit with INFO = 0 if swap was successfully performed. */

    return;

/*     Exit with INFO = 1 if swap was rejected. */

L20:
    *info = 1;
    return;

/*     End of CTGEX2 */

} /* ctgex2_ */

