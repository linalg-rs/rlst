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

static quadcomplex c_b1 = {1.,0.};
static integer c__1 = 1;
static integer c_n1 = -1;
static quadreal c_b24 = 1.;

/* > \brief \b ZLATDF uses the LU factorization of the n-by-n matrix computed by sgetc2 and computes a contrib
ution to the reciprocal Dif-estimate. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLATDF + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlatdf.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlatdf.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlatdf.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLATDF( IJOB, N, Z, LDZ, RHS, RDSUM, RDSCAL, IPIV, */
/*                          JPIV ) */

/*       INTEGER            IJOB, LDZ, N */
/*       DOUBLE PRECISION   RDSCAL, RDSUM */
/*       INTEGER            IPIV( * ), JPIV( * ) */
/*       COMPLEX*16         RHS( * ), Z( LDZ, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLATDF computes the contribution to the reciprocal Dif-estimate */
/* > by solving for x in Z * x = b, where b is chosen such that the norm */
/* > of x is as large as possible. It is assumed that LU decomposition */
/* > of Z has been computed by ZGETC2. On entry RHS = f holds the */
/* > contribution from earlier solved sub-systems, and on return RHS = x. */
/* > */
/* > The factorization of Z returned by ZGETC2 has the form */
/* > Z = P * L * U * Q, where P and Q are permutation matrices. L is lower */
/* > triangular with unit diagonal elements and U is upper triangular. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] IJOB */
/* > \verbatim */
/* >          IJOB is INTEGER */
/* >          IJOB = 2: First compute an approximative null-vector e */
/* >              of Z using ZGECON, e is normalized and solve for */
/* >              Zx = +-e - f with the sign giving the greater value of */
/* >              2-norm(x).  About 5 times as expensive as Default. */
/* >          IJOB .ne. 2: Local look ahead strategy where */
/* >              all entries of the r.h.s. b is chosen as either +1 or */
/* >              -1.  Default. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix Z. */
/* > \endverbatim */
/* > */
/* > \param[in] Z */
/* > \verbatim */
/* >          Z is COMPLEX*16 array, dimension (LDZ, N) */
/* >          On entry, the LU part of the factorization of the n-by-n */
/* >          matrix Z computed by ZGETC2:  Z = P * L * U * Q */
/* > \endverbatim */
/* > */
/* > \param[in] LDZ */
/* > \verbatim */
/* >          LDZ is INTEGER */
/* >          The leading dimension of the array Z.  LDA >= f2cmax(1, N). */
/* > \endverbatim */
/* > */
/* > \param[in,out] RHS */
/* > \verbatim */
/* >          RHS is COMPLEX*16 array, dimension (N). */
/* >          On entry, RHS contains contributions from other subsystems. */
/* >          On exit, RHS contains the solution of the subsystem with */
/* >          entries according to the value of IJOB (see above). */
/* > \endverbatim */
/* > */
/* > \param[in,out] RDSUM */
/* > \verbatim */
/* >          RDSUM is DOUBLE PRECISION */
/* >          On entry, the sum of squares of computed contributions to */
/* >          the Dif-estimate under computation by ZTGSYL, where the */
/* >          scaling factor RDSCAL (see below) has been factored out. */
/* >          On exit, the corresponding sum of squares updated with the */
/* >          contributions from the current sub-system. */
/* >          If TRANS = 'T' RDSUM is not touched. */
/* >          NOTE: RDSUM only makes sense when ZTGSY2 is called by CTGSYL. */
/* > \endverbatim */
/* > */
/* > \param[in,out] RDSCAL */
/* > \verbatim */
/* >          RDSCAL is DOUBLE PRECISION */
/* >          On entry, scaling factor used to prevent overflow in RDSUM. */
/* >          On exit, RDSCAL is updated w.r.t. the current contributions */
/* >          in RDSUM. */
/* >          If TRANS = 'T', RDSCAL is not touched. */
/* >          NOTE: RDSCAL only makes sense when ZTGSY2 is called by */
/* >          ZTGSYL. */
/* > \endverbatim */
/* > */
/* > \param[in] IPIV */
/* > \verbatim */
/* >          IPIV is INTEGER array, dimension (N). */
/* >          The pivot indices; for 1 <= i <= N, row i of the */
/* >          matrix has been interchanged with row IPIV(i). */
/* > \endverbatim */
/* > */
/* > \param[in] JPIV */
/* > \verbatim */
/* >          JPIV is INTEGER array, dimension (N). */
/* >          The pivot indices; for 1 <= j <= N, column j of the */
/* >          matrix has been interchanged with column JPIV(j). */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2016 */

/* > \ingroup complex16OTHERauxiliary */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* >  This routine is a further developed implementation of algorithm */
/* >  BSOLVE in [1] using complete pivoting in the LU factorization. */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >     Bo Kagstrom and Peter Poromaa, Department of Computing Science, */
/* >     Umea University, S-901 87 Umea, Sweden. */

/* > \par References: */
/*  ================ */
/* > */
/* >   [1]   Bo Kagstrom and Lars Westin, */
/* >         Generalized Schur Methods with Condition Estimators for */
/* >         Solving the Generalized Sylvester Equation, IEEE Transactions */
/* >         on Automatic Control, Vol. 34, No. 7, July 1989, pp 745-751. */
/* >\n */
/* >   [2]   Peter Poromaa, */
/* >         On Efficient and Robust Estimators for the Separation */
/* >         between two Regular Matrix Pairs with Applications in */
/* >         Condition Estimation. Report UMINF-95.05, Department of */
/* >         Computing Science, Umea University, S-901 87 Umea, Sweden, */
/* >         1995. */

/*  ===================================================================== */
void  wlatdf_(integer *ijob, integer *n, quadcomplex *z__, 
	integer *ldz, quadcomplex *rhs, quadreal *rdsum, quadreal *
	rdscal, integer *ipiv, integer *jpiv)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1, i__2, i__3, i__4, i__5;
    quadcomplex z__1, z__2, z__3;

    /* Local variables */
    integer i__, j, k;
    quadcomplex bm, bp, xm[2], xp[2];
    integer info;
    quadcomplex temp, work[8];
    quadreal scale;
    extern void  wscal_(integer *, quadcomplex *, 
	    quadcomplex *, integer *);
    quadcomplex pmone;
    extern /* Double Complex */ VOID wqotc_(quadcomplex *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, integer *);
    quadreal rtemp, sminu, rwork[2];
    extern void  wcopy_(integer *, quadcomplex *, integer *, 
	    quadcomplex *, integer *);
    quadreal splus;
    extern void  waxpy_(integer *, quadcomplex *, 
	    quadcomplex *, integer *, quadcomplex *, integer *), wgesc2_(
	    integer *, quadcomplex *, integer *, quadcomplex *, integer *,
	     integer *, quadreal *), wgecon_(char *, integer *, 
	    quadcomplex *, integer *, quadreal *, quadreal *, 
	    quadcomplex *, quadreal *, integer *);
    extern quadreal qwasum_(integer *, quadcomplex *, integer *);
    extern void  wlassq_(integer *, quadcomplex *, integer *,
	     quadreal *, quadreal *), wlaswp_(integer *, quadcomplex *, 
	    integer *, integer *, integer *, integer *, integer *);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --rhs;
    --ipiv;
    --jpiv;

    /* Function Body */
    if (*ijob != 2) {

/*        Apply permutations IPIV to RHS */

	i__1 = *n - 1;
	wlaswp_(&c__1, &rhs[1], ldz, &c__1, &i__1, &ipiv[1], &c__1);

/*        Solve for L-part choosing RHS either to +1 or -1. */

	z__1.r = -1., z__1.i = -0.;
	pmone.r = z__1.r, pmone.i = z__1.i;
	i__1 = *n - 1;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j;
	    z__1.r = rhs[i__2].r + 1., z__1.i = rhs[i__2].i + 0.;
	    bp.r = z__1.r, bp.i = z__1.i;
	    i__2 = j;
	    z__1.r = rhs[i__2].r - 1., z__1.i = rhs[i__2].i - 0.;
	    bm.r = z__1.r, bm.i = z__1.i;
	    splus = 1.;

/*           Lockahead for L- part RHS(1:N-1) = +-1 */
/*           SPLUS and SMIN computed more efficiently than in BSOLVE[1]. */

	    i__2 = *n - j;
	    wqotc_(&z__1, &i__2, &z__[j + 1 + j * z_dim1], &c__1, &z__[j + 1 
		    + j * z_dim1], &c__1);
	    splus += z__1.r;
	    i__2 = *n - j;
	    wqotc_(&z__1, &i__2, &z__[j + 1 + j * z_dim1], &c__1, &rhs[j + 1],
		     &c__1);
	    sminu = z__1.r;
	    i__2 = j;
	    splus *= rhs[i__2].r;
	    if (splus > sminu) {
		i__2 = j;
		rhs[i__2].r = bp.r, rhs[i__2].i = bp.i;
	    } else if (sminu > splus) {
		i__2 = j;
		rhs[i__2].r = bm.r, rhs[i__2].i = bm.i;
	    } else {

/*              In this case the updating sums are equal and we can */
/*              choose RHS(J) +1 or -1. The first time this happens we */
/*              choose -1, thereafter +1. This is a simple way to get */
/*              good estimates of matrices like Byers well-known example */
/*              (see [1]). (Not done in BSOLVE.) */

		i__2 = j;
		i__3 = j;
		z__1.r = rhs[i__3].r + pmone.r, z__1.i = rhs[i__3].i + 
			pmone.i;
		rhs[i__2].r = z__1.r, rhs[i__2].i = z__1.i;
		pmone.r = 1., pmone.i = 0.;
	    }

/*           Compute the remaining r.h.s. */

	    i__2 = j;
	    z__1.r = -rhs[i__2].r, z__1.i = -rhs[i__2].i;
	    temp.r = z__1.r, temp.i = z__1.i;
	    i__2 = *n - j;
	    waxpy_(&i__2, &temp, &z__[j + 1 + j * z_dim1], &c__1, &rhs[j + 1],
		     &c__1);
/* L10: */
	}

/*        Solve for U- part, lockahead for RHS(N) = +-1. This is not done */
/*        In BSOLVE and will hopefully give us a better estimate because */
/*        any ill-conditioning of the original matrix is transfered to U */
/*        and not to L. U(N, N) is an approximation to sigma_min(LU). */

	i__1 = *n - 1;
	wcopy_(&i__1, &rhs[1], &c__1, work, &c__1);
	i__1 = *n - 1;
	i__2 = *n;
	z__1.r = rhs[i__2].r + 1., z__1.i = rhs[i__2].i + 0.;
	work[i__1].r = z__1.r, work[i__1].i = z__1.i;
	i__1 = *n;
	i__2 = *n;
	z__1.r = rhs[i__2].r - 1., z__1.i = rhs[i__2].i - 0.;
	rhs[i__1].r = z__1.r, rhs[i__1].i = z__1.i;
	splus = 0.;
	sminu = 0.;
	for (i__ = *n; i__ >= 1; --i__) {
	    z_div(&z__1, &c_b1, &z__[i__ + i__ * z_dim1]);
	    temp.r = z__1.r, temp.i = z__1.i;
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    z__1.r = work[i__2].r * temp.r - work[i__2].i * temp.i, z__1.i = 
		    work[i__2].r * temp.i + work[i__2].i * temp.r;
	    work[i__1].r = z__1.r, work[i__1].i = z__1.i;
	    i__1 = i__;
	    i__2 = i__;
	    z__1.r = rhs[i__2].r * temp.r - rhs[i__2].i * temp.i, z__1.i = 
		    rhs[i__2].r * temp.i + rhs[i__2].i * temp.r;
	    rhs[i__1].r = z__1.r, rhs[i__1].i = z__1.i;
	    i__1 = *n;
	    for (k = i__ + 1; k <= i__1; ++k) {
		i__2 = i__ - 1;
		i__3 = i__ - 1;
		i__4 = k - 1;
		i__5 = i__ + k * z_dim1;
		z__3.r = z__[i__5].r * temp.r - z__[i__5].i * temp.i, z__3.i =
			 z__[i__5].r * temp.i + z__[i__5].i * temp.r;
		z__2.r = work[i__4].r * z__3.r - work[i__4].i * z__3.i, 
			z__2.i = work[i__4].r * z__3.i + work[i__4].i * 
			z__3.r;
		z__1.r = work[i__3].r - z__2.r, z__1.i = work[i__3].i - 
			z__2.i;
		work[i__2].r = z__1.r, work[i__2].i = z__1.i;
		i__2 = i__;
		i__3 = i__;
		i__4 = k;
		i__5 = i__ + k * z_dim1;
		z__3.r = z__[i__5].r * temp.r - z__[i__5].i * temp.i, z__3.i =
			 z__[i__5].r * temp.i + z__[i__5].i * temp.r;
		z__2.r = rhs[i__4].r * z__3.r - rhs[i__4].i * z__3.i, z__2.i =
			 rhs[i__4].r * z__3.i + rhs[i__4].i * z__3.r;
		z__1.r = rhs[i__3].r - z__2.r, z__1.i = rhs[i__3].i - z__2.i;
		rhs[i__2].r = z__1.r, rhs[i__2].i = z__1.i;
/* L20: */
	    }
	    splus += z_abs(&work[i__ - 1]);
	    sminu += z_abs(&rhs[i__]);
/* L30: */
	}
	if (splus > sminu) {
	    wcopy_(n, work, &c__1, &rhs[1], &c__1);
	}

/*        Apply the permutations JPIV to the computed solution (RHS) */

	i__1 = *n - 1;
	wlaswp_(&c__1, &rhs[1], ldz, &c__1, &i__1, &jpiv[1], &c_n1);

/*        Compute the sum of squares */

	wlassq_(n, &rhs[1], &c__1, rdscal, rdsum);
	return;
    }

/*     ENTRY IJOB = 2 */

/*     Compute approximate nullvector XM of Z */

    wgecon_("I", n, &z__[z_offset], ldz, &c_b24, &rtemp, work, rwork, &info);
    wcopy_(n, &work[*n], &c__1, xm, &c__1);

/*     Compute RHS */

    i__1 = *n - 1;
    wlaswp_(&c__1, xm, ldz, &c__1, &i__1, &ipiv[1], &c_n1);
    wqotc_(&z__3, n, xm, &c__1, xm, &c__1);
    z_sqrt(&z__2, &z__3);
    z_div(&z__1, &c_b1, &z__2);
    temp.r = z__1.r, temp.i = z__1.i;
    wscal_(n, &temp, xm, &c__1);
    wcopy_(n, xm, &c__1, xp, &c__1);
    waxpy_(n, &c_b1, &rhs[1], &c__1, xp, &c__1);
    z__1.r = -1., z__1.i = -0.;
    waxpy_(n, &z__1, xm, &c__1, &rhs[1], &c__1);
    wgesc2_(n, &z__[z_offset], ldz, &rhs[1], &ipiv[1], &jpiv[1], &scale);
    wgesc2_(n, &z__[z_offset], ldz, xp, &ipiv[1], &jpiv[1], &scale);
    if (qwasum_(n, xp, &c__1) > qwasum_(n, &rhs[1], &c__1)) {
	wcopy_(n, xp, &c__1, &rhs[1], &c__1);
    }

/*     Compute the sum of squares */

    wlassq_(n, &rhs[1], &c__1, rdscal, rdsum);
    return;

/*     End of ZLATDF */

} /* wlatdf_ */

