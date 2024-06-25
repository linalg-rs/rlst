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

static integer c__1 = 1;
static integer c__0 = 0;
static real c_b27 = 1.f;

/* > \brief \b CGSVJ0 pre-processor for the routine cgesvj. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CGSVJ0 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/cgsvj0.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/cgsvj0.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/cgsvj0.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CGSVJ0( JOBV, M, N, A, LDA, D, SVA, MV, V, LDV, EPS, */
/*                          SFMIN, TOL, NSWEEP, WORK, LWORK, INFO ) */

/*       INTEGER            INFO, LDA, LDV, LWORK, M, MV, N, NSWEEP */
/*       REAL               EPS, SFMIN, TOL */
/*       CHARACTER*1        JOBV */
/*       COMPLEX            A( LDA, * ), D( N ), V( LDV, * ), WORK( LWORK ) */
/*       REAL               SVA( N ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CGSVJ0 is called from CGESVJ as a pre-processor and that is its main */
/* > purpose. It applies Jacobi rotations in the same way as CGESVJ does, but */
/* > it does not check convergence (stopping criterion). Few tuning */
/* > parameters (marked by [TP]) are available for the implementer. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] JOBV */
/* > \verbatim */
/* >          JOBV is CHARACTER*1 */
/* >          Specifies whether the output from this procedure is used */
/* >          to compute the matrix V: */
/* >          = 'V': the product of the Jacobi rotations is accumulated */
/* >                 by postmulyiplying the N-by-N array V. */
/* >                (See the description of V.) */
/* >          = 'A': the product of the Jacobi rotations is accumulated */
/* >                 by postmulyiplying the MV-by-N array V. */
/* >                (See the descriptions of MV and V.) */
/* >          = 'N': the Jacobi rotations are not accumulated. */
/* > \endverbatim */
/* > */
/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the input matrix A.  M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the input matrix A. */
/* >          M >= N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX array, dimension (LDA,N) */
/* >          On entry, M-by-N matrix A, such that A*diag(D) represents */
/* >          the input matrix. */
/* >          On exit, */
/* >          A_onexit * diag(D_onexit) represents the input matrix A*diag(D) */
/* >          post-multiplied by a sequence of Jacobi rotations, where the */
/* >          rotation threshold and the total number of sweeps are given in */
/* >          TOL and NSWEEP, respectively. */
/* >          (See the descriptions of D, TOL and NSWEEP.) */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[in,out] D */
/* > \verbatim */
/* >          D is COMPLEX array, dimension (N) */
/* >          The array D accumulates the scaling factors from the complex scaled */
/* >          Jacobi rotations. */
/* >          On entry, A*diag(D) represents the input matrix. */
/* >          On exit, A_onexit*diag(D_onexit) represents the input matrix */
/* >          post-multiplied by a sequence of Jacobi rotations, where the */
/* >          rotation threshold and the total number of sweeps are given in */
/* >          TOL and NSWEEP, respectively. */
/* >          (See the descriptions of A, TOL and NSWEEP.) */
/* > \endverbatim */
/* > */
/* > \param[in,out] SVA */
/* > \verbatim */
/* >          SVA is REAL array, dimension (N) */
/* >          On entry, SVA contains the Euclidean norms of the columns of */
/* >          the matrix A*diag(D). */
/* >          On exit, SVA contains the Euclidean norms of the columns of */
/* >          the matrix A_onexit*diag(D_onexit). */
/* > \endverbatim */
/* > */
/* > \param[in] MV */
/* > \verbatim */
/* >          MV is INTEGER */
/* >          If JOBV .EQ. 'A', then MV rows of V are post-multipled by a */
/* >                           sequence of Jacobi rotations. */
/* >          If JOBV = 'N',   then MV is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in,out] V */
/* > \verbatim */
/* >          V is COMPLEX array, dimension (LDV,N) */
/* >          If JOBV .EQ. 'V' then N rows of V are post-multipled by a */
/* >                           sequence of Jacobi rotations. */
/* >          If JOBV .EQ. 'A' then MV rows of V are post-multipled by a */
/* >                           sequence of Jacobi rotations. */
/* >          If JOBV = 'N',   then V is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDV */
/* > \verbatim */
/* >          LDV is INTEGER */
/* >          The leading dimension of the array V,  LDV >= 1. */
/* >          If JOBV = 'V', LDV .GE. N. */
/* >          If JOBV = 'A', LDV .GE. MV. */
/* > \endverbatim */
/* > */
/* > \param[in] EPS */
/* > \verbatim */
/* >          EPS is REAL */
/* >          EPS = SLAMCH('Epsilon') */
/* > \endverbatim */
/* > */
/* > \param[in] SFMIN */
/* > \verbatim */
/* >          SFMIN is REAL */
/* >          SFMIN = SLAMCH('Safe Minimum') */
/* > \endverbatim */
/* > */
/* > \param[in] TOL */
/* > \verbatim */
/* >          TOL is REAL */
/* >          TOL is the threshold for Jacobi rotations. For a pair */
/* >          A(:,p), A(:,q) of pivot columns, the Jacobi rotation is */
/* >          applied only if ABS(COS(angle(A(:,p),A(:,q)))) .GT. TOL. */
/* > \endverbatim */
/* > */
/* > \param[in] NSWEEP */
/* > \verbatim */
/* >          NSWEEP is INTEGER */
/* >          NSWEEP is the number of sweeps of Jacobi rotations to be */
/* >          performed. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX array, dimension (LWORK) */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          LWORK is the dimension of WORK. LWORK .GE. M. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0 : successful exit. */
/* >          < 0 : if INFO = -i, then the i-th argument had an illegal value */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2016 */

/* > \ingroup complexOTHERcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > CGSVJ0 is used just to enable CGESVJ to call a simplified version of */
/* > itself to work on a submatrix of the original matrix. */
/* > */
/* > \par Contributor: */
/*  ================== */
/* > */
/* > Zlatko Drmac (Zagreb, Croatia) */
/* > */
/* > \par Bugs, Examples and Comments: */
/*  ================================= */
/* > */
/* > Please report all bugs and send interesting test examples and comments to */
/* > drmac@math.hr. Thank you. */

/*  ===================================================================== */
void  cgsvj0_(char *jobv, integer *m, integer *n, complex *a, 
	integer *lda, complex *d__, real *sva, integer *mv, complex *v, 
	integer *ldv, real *eps, real *sfmin, real *tol, integer *nsweep, 
	complex *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, v_dim1, v_offset, i__1, i__2, i__3, i__4, i__5, 
	    i__6, i__7;
    real r__1, r__2;
    complex q__1, q__2, q__3;

    /* Local variables */
    real bigtheta;
    integer pskipped, i__, p, q;
    real t, rootsfmin, cs, sn;
    integer ir1, jbc;
    real big;
    integer kbl, igl, ibr, jgl, nbl, mvl;
    real aapp;
    complex aapq;
    real aaqq;
    integer ierr;
    extern void  crot_(integer *, complex *, integer *, 
	    complex *, integer *, real *, complex *);
    complex ompq;
    real aapp0, aapq1, temp1;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    real apoaq, aqoap;
    extern logical lsame_(char *, char *);
    real theta, small;
    extern void  ccopy_(integer *, complex *, integer *, 
	    complex *, integer *), cswap_(integer *, complex *, integer *, 
	    complex *, integer *);
    logical applv, rsvec;
    extern void  caxpy_(integer *, complex *, complex *, 
	    integer *, complex *, integer *);
    logical rotok;
    extern real scnrm2_(integer *, complex *, integer *);
    extern void  clascl_(char *, integer *, integer *, real *, 
	    real *, integer *, integer *, complex *, integer *, integer *), xerbla_(char *, integer *);
    integer ijblsk, swband;
    extern integer isamax_(integer *, real *, integer *);
    integer blskip;
    extern void  classq_(integer *, complex *, integer *, real 
	    *, real *);
    real mxaapq, thsign, mxsinj;
    integer emptsw, notrot, iswrot, lkahead;
    real rootbig, rooteps;
    integer rowskip;
    real roottol;


/*  -- LAPACK computational routine (version 3.8.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2016 */


/*  ===================================================================== */

/*     from BLAS */
/*     from LAPACK */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --sva;
    --d__;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    --work;

    /* Function Body */
    applv = lsame_(jobv, "A");
    rsvec = lsame_(jobv, "V");
    if (! (rsvec || applv || lsame_(jobv, "N"))) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (*n < 0 || *n > *m) {
	*info = -3;
    } else if (*lda < *m) {
	*info = -5;
    } else if ((rsvec || applv) && *mv < 0) {
	*info = -8;
    } else if (rsvec && *ldv < *n || applv && *ldv < *mv) {
	*info = -10;
    } else if (*tol <= *eps) {
	*info = -13;
    } else if (*nsweep < 0) {
	*info = -14;
    } else if (*lwork < *m) {
	*info = -16;
    } else {
	*info = 0;
    }

/*     #:( */
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CGSVJ0", &i__1);
	return;
    }

    if (rsvec) {
	mvl = *n;
    } else if (applv) {
	mvl = *mv;
    }
    rsvec = rsvec || applv;
    rooteps = M(sqrt)(*eps);
    rootsfmin = M(sqrt)(*sfmin);
    small = *sfmin / *eps;
    big = 1.f / *sfmin;
    rootbig = 1.f / rootsfmin;
    bigtheta = 1.f / rooteps;
    roottol = M(sqrt)(*tol);


    emptsw = *n * (*n - 1) / 2;
    notrot = 0;


    swband = 0;
/* [TP] SWBAND is a tuning parameter [TP]. It is meaningful and effective */
/*     if CGESVJ is used as a computational routine in the preconditioned */
/*     Jacobi SVD algorithm CGEJSV. For sweeps i=1:SWBAND the procedure */
/*     works on pivots inside a band-like region around the diagonal. */
/*     The boundaries are determined dynamically, based on the number of */
/*     pivots above a threshold. */

    kbl = f2cmin(8,*n);
/* [TP] KBL is a tuning parameter that defines the tile size in the */
/*     tiling of the p-q loops of pivot pairs. In general, an optimal */
/*     value of KBL depends on the matrix dimensions and on the */
/*     parameters of the computer's memory. */

    nbl = *n / kbl;
    if (nbl * kbl != *n) {
	++nbl;
    }

/* Computing 2nd power */
    i__1 = kbl;
    blskip = i__1 * i__1;
/* [TP] BLKSKIP is a tuning parameter that depends on SWBAND and KBL. */

    rowskip = f2cmin(5,kbl);
/* [TP] ROWSKIP is a tuning parameter. */

    lkahead = 1;
/* [TP] LKAHEAD is a tuning parameter. */

/*     Quasi block transformations, using the lower (upper) triangular */
/*     structure of the input matrix. The quasi-block-cycling usually */
/*     invokes cubic convergence. Big part of this cycle is done inside */
/*     canonical subspaces of dimensions less than M. */



    i__1 = *nsweep;
    for (i__ = 1; i__ <= i__1; ++i__) {


	mxaapq = 0.f;
	mxsinj = 0.f;
	iswrot = 0;

	notrot = 0;
	pskipped = 0;

/*     Each sweep is unrolled using KBL-by-KBL tiles over the pivot pairs */
/*     1 <= p < q <= N. This is the first step toward a blocked implementation */
/*     of the rotations. New implementation, based on block transformations, */
/*     is under development. */

	i__2 = nbl;
	for (ibr = 1; ibr <= i__2; ++ibr) {

	    igl = (ibr - 1) * kbl + 1;

/* Computing MIN */
	    i__4 = lkahead, i__5 = nbl - ibr;
	    i__3 = f2cmin(i__4,i__5);
	    for (ir1 = 0; ir1 <= i__3; ++ir1) {

		igl += ir1 * kbl;

/* Computing MIN */
		i__5 = igl + kbl - 1, i__6 = *n - 1;
		i__4 = f2cmin(i__5,i__6);
		for (p = igl; p <= i__4; ++p) {


		    i__5 = *n - p + 1;
		    q = isamax_(&i__5, &sva[p], &c__1) + p - 1;
		    if (p != q) {
			cswap_(m, &a[p * a_dim1 + 1], &c__1, &a[q * a_dim1 + 
				1], &c__1);
			if (rsvec) {
			    cswap_(&mvl, &v[p * v_dim1 + 1], &c__1, &v[q * 
				    v_dim1 + 1], &c__1);
			}
			temp1 = sva[p];
			sva[p] = sva[q];
			sva[q] = temp1;
			i__5 = p;
			aapq.r = d__[i__5].r, aapq.i = d__[i__5].i;
			i__5 = p;
			i__6 = q;
			d__[i__5].r = d__[i__6].r, d__[i__5].i = d__[i__6].i;
			i__5 = q;
			d__[i__5].r = aapq.r, d__[i__5].i = aapq.i;
		    }

		    if (ir1 == 0) {

/*        Column norms are periodically updated by explicit */
/*        norm computation. */
/*        Caveat: */
/*        Unfortunately, some BLAS implementations compute SNCRM2(M,A(1,p),1) */
/*        as SQRT(S=CDOTC(M,A(1,p),1,A(1,p),1)), which may cause the result to */
/*        overflow for ||A(:,p)||_2 > SQRT(overflow_threshold), and to */
/*        underflow for ||A(:,p)||_2 < SQRT(underflow_threshold). */
/*        Hence, SCNRM2 cannot be trusted, not even in the case when */
/*        the true norm is far from the under(over)flow boundaries. */
/*        If properly implemented SCNRM2 is available, the IF-THEN-ELSE-END IF */
/*        below should be replaced with "AAPP = SCNRM2( M, A(1,p), 1 )". */

			if (sva[p] < rootbig && sva[p] > rootsfmin) {
			    sva[p] = scnrm2_(m, &a[p * a_dim1 + 1], &c__1);
			} else {
			    temp1 = 0.f;
			    aapp = 1.f;
			    classq_(m, &a[p * a_dim1 + 1], &c__1, &temp1, &
				    aapp);
			    sva[p] = temp1 * M(sqrt)(aapp);
			}
			aapp = sva[p];
		    } else {
			aapp = sva[p];
		    }

		    if (aapp > 0.f) {

			pskipped = 0;

/* Computing MIN */
			i__6 = igl + kbl - 1;
			i__5 = f2cmin(i__6,*n);
			for (q = p + 1; q <= i__5; ++q) {

			    aaqq = sva[q];

			    if (aaqq > 0.f) {

				aapp0 = aapp;
				if (aaqq >= 1.f) {
				    rotok = small * aapp <= aaqq;
				    if (aapp < big / aaqq) {
					cdotc_(&q__3, m, &a[p * a_dim1 + 1], &
						c__1, &a[q * a_dim1 + 1], &
						c__1);
					q__2.r = q__3.r / aaqq, q__2.i = 
						q__3.i / aaqq;
					q__1.r = q__2.r / aapp, q__1.i = 
						q__2.i / aapp;
					aapq.r = q__1.r, aapq.i = q__1.i;
				    } else {
					ccopy_(m, &a[p * a_dim1 + 1], &c__1, &
						work[1], &c__1);
					clascl_("G", &c__0, &c__0, &aapp, &
						c_b27, m, &c__1, &work[1], 
						lda, &ierr);
					cdotc_(&q__2, m, &work[1], &c__1, &a[
						q * a_dim1 + 1], &c__1);
					q__1.r = q__2.r / aaqq, q__1.i = 
						q__2.i / aaqq;
					aapq.r = q__1.r, aapq.i = q__1.i;
				    }
				} else {
				    rotok = aapp <= aaqq / small;
				    if (aapp > small / aaqq) {
					cdotc_(&q__3, m, &a[p * a_dim1 + 1], &
						c__1, &a[q * a_dim1 + 1], &
						c__1);
					q__2.r = q__3.r / aapp, q__2.i = 
						q__3.i / aapp;
					q__1.r = q__2.r / aaqq, q__1.i = 
						q__2.i / aaqq;
					aapq.r = q__1.r, aapq.i = q__1.i;
				    } else {
					ccopy_(m, &a[q * a_dim1 + 1], &c__1, &
						work[1], &c__1);
					clascl_("G", &c__0, &c__0, &aaqq, &
						c_b27, m, &c__1, &work[1], 
						lda, &ierr);
					cdotc_(&q__2, m, &a[p * a_dim1 + 1], &
						c__1, &work[1], &c__1);
					q__1.r = q__2.r / aapp, q__1.i = 
						q__2.i / aapp;
					aapq.r = q__1.r, aapq.i = q__1.i;
				    }
				}

/*                           AAPQ = AAPQ * CONJG( CWORK(p) ) * CWORK(q) */
				aapq1 = -c_abs(&aapq);
/* Computing MAX */
				r__1 = mxaapq, r__2 = -aapq1;
				mxaapq = f2cmax(r__1,r__2);

/*        TO rotate or NOT to rotate, THAT is the question ... */

				if (abs(aapq1) > *tol) {
				    r__1 = c_abs(&aapq);
				    q__1.r = aapq.r / r__1, q__1.i = aapq.i / 
					    r__1;
				    ompq.r = q__1.r, ompq.i = q__1.i;

/* [RTD]      ROTATED = ROTATED + ONE */

				    if (ir1 == 0) {
					notrot = 0;
					pskipped = 0;
					++iswrot;
				    }

				    if (rotok) {

					aqoap = aaqq / aapp;
					apoaq = aapp / aaqq;
					theta = (r__1 = aqoap - apoaq, abs(
						r__1)) * -.5f / aapq1;

					if (abs(theta) > bigtheta) {

					    t = .5f / theta;
					    cs = 1.f;
					    r_cnjg(&q__2, &ompq);
					    q__1.r = t * q__2.r, q__1.i = t * 
						    q__2.i;
					    crot_(m, &a[p * a_dim1 + 1], &
						    c__1, &a[q * a_dim1 + 1], 
						    &c__1, &cs, &q__1);
					    if (rsvec) {
			  r_cnjg(&q__2, &ompq);
			  q__1.r = t * q__2.r, q__1.i = t * q__2.i;
			  crot_(&mvl, &v[p * v_dim1 + 1], &c__1, &v[q * 
				  v_dim1 + 1], &c__1, &cs, &q__1);
					    }
/* Computing MAX */
					    r__1 = 0.f, r__2 = t * apoaq * 
						    aapq1 + 1.f;
					    sva[q] = aaqq * M(sqrt)((f2cmax(r__1,
						    r__2)));
/* Computing MAX */
					    r__1 = 0.f, r__2 = 1.f - t * 
						    aqoap * aapq1;
					    aapp *= M(sqrt)((f2cmax(r__1,r__2)));
/* Computing MAX */
					    r__1 = mxsinj, r__2 = abs(t);
					    mxsinj = f2cmax(r__1,r__2);

					} else {


					    thsign = -r_sign(&c_b27, &aapq1);
					    t = 1.f / (theta + thsign * M(sqrt)(
						    theta * theta + 1.f));
					    cs = M(sqrt)(1.f / (t * t + 1.f));
					    sn = t * cs;

/* Computing MAX */
					    r__1 = mxsinj, r__2 = abs(sn);
					    mxsinj = f2cmax(r__1,r__2);
/* Computing MAX */
					    r__1 = 0.f, r__2 = t * apoaq * 
						    aapq1 + 1.f;
					    sva[q] = aaqq * M(sqrt)((f2cmax(r__1,
						    r__2)));
/* Computing MAX */
					    r__1 = 0.f, r__2 = 1.f - t * 
						    aqoap * aapq1;
					    aapp *= M(sqrt)((f2cmax(r__1,r__2)));

					    r_cnjg(&q__2, &ompq);
					    q__1.r = sn * q__2.r, q__1.i = sn 
						    * q__2.i;
					    crot_(m, &a[p * a_dim1 + 1], &
						    c__1, &a[q * a_dim1 + 1], 
						    &c__1, &cs, &q__1);
					    if (rsvec) {
			  r_cnjg(&q__2, &ompq);
			  q__1.r = sn * q__2.r, q__1.i = sn * q__2.i;
			  crot_(&mvl, &v[p * v_dim1 + 1], &c__1, &v[q * 
				  v_dim1 + 1], &c__1, &cs, &q__1);
					    }
					}
					i__6 = p;
					i__7 = q;
					q__2.r = -d__[i__7].r, q__2.i = -d__[
						i__7].i;
					q__1.r = q__2.r * ompq.r - q__2.i * 
						ompq.i, q__1.i = q__2.r * 
						ompq.i + q__2.i * ompq.r;
					d__[i__6].r = q__1.r, d__[i__6].i = 
						q__1.i;

				    } else {
					ccopy_(m, &a[p * a_dim1 + 1], &c__1, &
						work[1], &c__1);
					clascl_("G", &c__0, &c__0, &aapp, &
						c_b27, m, &c__1, &work[1], 
						lda, &ierr);
					clascl_("G", &c__0, &c__0, &aaqq, &
						c_b27, m, &c__1, &a[q * 
						a_dim1 + 1], lda, &ierr);
					q__1.r = -aapq.r, q__1.i = -aapq.i;
					caxpy_(m, &q__1, &work[1], &c__1, &a[
						q * a_dim1 + 1], &c__1);
					clascl_("G", &c__0, &c__0, &c_b27, &
						aaqq, m, &c__1, &a[q * a_dim1 
						+ 1], lda, &ierr);
/* Computing MAX */
					r__1 = 0.f, r__2 = 1.f - aapq1 * 
						aapq1;
					sva[q] = aaqq * M(sqrt)((f2cmax(r__1,r__2)))
						;
					mxsinj = f2cmax(mxsinj,*sfmin);
				    }
/*           END IF ROTOK THEN ... ELSE */

/*           In the case of cancellation in updating SVA(q), SVA(p) */
/*           recompute SVA(q), SVA(p). */

/* Computing 2nd power */
				    r__1 = sva[q] / aaqq;
				    if (r__1 * r__1 <= rooteps) {
					if (aaqq < rootbig && aaqq > 
						rootsfmin) {
					    sva[q] = scnrm2_(m, &a[q * a_dim1 
						    + 1], &c__1);
					} else {
					    t = 0.f;
					    aaqq = 1.f;
					    classq_(m, &a[q * a_dim1 + 1], &
						    c__1, &t, &aaqq);
					    sva[q] = t * M(sqrt)(aaqq);
					}
				    }
				    if (aapp / aapp0 <= rooteps) {
					if (aapp < rootbig && aapp > 
						rootsfmin) {
					    aapp = scnrm2_(m, &a[p * a_dim1 + 
						    1], &c__1);
					} else {
					    t = 0.f;
					    aapp = 1.f;
					    classq_(m, &a[p * a_dim1 + 1], &
						    c__1, &t, &aapp);
					    aapp = t * M(sqrt)(aapp);
					}
					sva[p] = aapp;
				    }

				} else {
/*        A(:,p) and A(:,q) already numerically orthogonal */
				    if (ir1 == 0) {
					++notrot;
				    }
/* [RTD]      SKIPPED  = SKIPPED  + 1 */
				    ++pskipped;
				}
			    } else {
/*        A(:,q) is zero column */
				if (ir1 == 0) {
				    ++notrot;
				}
				++pskipped;
			    }

			    if (i__ <= swband && pskipped > rowskip) {
				if (ir1 == 0) {
				    aapp = -aapp;
				}
				notrot = 0;
				goto L2103;
			    }

/* L2002: */
			}
/*     END q-LOOP */

L2103:
/*     bailed out of q-loop */

			sva[p] = aapp;

		    } else {
			sva[p] = aapp;
			if (ir1 == 0 && aapp == 0.f) {
/* Computing MIN */
			    i__5 = igl + kbl - 1;
			    notrot = notrot + f2cmin(i__5,*n) - p;
			}
		    }

/* L2001: */
		}
/*     end of the p-loop */
/*     end of doing the block ( ibr, ibr ) */
/* L1002: */
	    }
/*     end of ir1-loop */

/* ... go to the off diagonal blocks */

	    igl = (ibr - 1) * kbl + 1;

	    i__3 = nbl;
	    for (jbc = ibr + 1; jbc <= i__3; ++jbc) {

		jgl = (jbc - 1) * kbl + 1;

/*        doing the block at ( ibr, jbc ) */

		ijblsk = 0;
/* Computing MIN */
		i__5 = igl + kbl - 1;
		i__4 = f2cmin(i__5,*n);
		for (p = igl; p <= i__4; ++p) {

		    aapp = sva[p];
		    if (aapp > 0.f) {

			pskipped = 0;

/* Computing MIN */
			i__6 = jgl + kbl - 1;
			i__5 = f2cmin(i__6,*n);
			for (q = jgl; q <= i__5; ++q) {

			    aaqq = sva[q];
			    if (aaqq > 0.f) {
				aapp0 = aapp;


/*        Safe Gram matrix computation */

				if (aaqq >= 1.f) {
				    if (aapp >= aaqq) {
					rotok = small * aapp <= aaqq;
				    } else {
					rotok = small * aaqq <= aapp;
				    }
				    if (aapp < big / aaqq) {
					cdotc_(&q__3, m, &a[p * a_dim1 + 1], &
						c__1, &a[q * a_dim1 + 1], &
						c__1);
					q__2.r = q__3.r / aaqq, q__2.i = 
						q__3.i / aaqq;
					q__1.r = q__2.r / aapp, q__1.i = 
						q__2.i / aapp;
					aapq.r = q__1.r, aapq.i = q__1.i;
				    } else {
					ccopy_(m, &a[p * a_dim1 + 1], &c__1, &
						work[1], &c__1);
					clascl_("G", &c__0, &c__0, &aapp, &
						c_b27, m, &c__1, &work[1], 
						lda, &ierr);
					cdotc_(&q__2, m, &work[1], &c__1, &a[
						q * a_dim1 + 1], &c__1);
					q__1.r = q__2.r / aaqq, q__1.i = 
						q__2.i / aaqq;
					aapq.r = q__1.r, aapq.i = q__1.i;
				    }
				} else {
				    if (aapp >= aaqq) {
					rotok = aapp <= aaqq / small;
				    } else {
					rotok = aaqq <= aapp / small;
				    }
				    if (aapp > small / aaqq) {
					cdotc_(&q__3, m, &a[p * a_dim1 + 1], &
						c__1, &a[q * a_dim1 + 1], &
						c__1);
					r__1 = f2cmax(aaqq,aapp);
					q__2.r = q__3.r / r__1, q__2.i = 
						q__3.i / r__1;
					r__2 = f2cmin(aaqq,aapp);
					q__1.r = q__2.r / r__2, q__1.i = 
						q__2.i / r__2;
					aapq.r = q__1.r, aapq.i = q__1.i;
				    } else {
					ccopy_(m, &a[q * a_dim1 + 1], &c__1, &
						work[1], &c__1);
					clascl_("G", &c__0, &c__0, &aaqq, &
						c_b27, m, &c__1, &work[1], 
						lda, &ierr);
					cdotc_(&q__2, m, &a[p * a_dim1 + 1], &
						c__1, &work[1], &c__1);
					q__1.r = q__2.r / aapp, q__1.i = 
						q__2.i / aapp;
					aapq.r = q__1.r, aapq.i = q__1.i;
				    }
				}

/*                           AAPQ = AAPQ * CONJG(CWORK(p))*CWORK(q) */
				aapq1 = -c_abs(&aapq);
/* Computing MAX */
				r__1 = mxaapq, r__2 = -aapq1;
				mxaapq = f2cmax(r__1,r__2);

/*        TO rotate or NOT to rotate, THAT is the question ... */

				if (abs(aapq1) > *tol) {
				    r__1 = c_abs(&aapq);
				    q__1.r = aapq.r / r__1, q__1.i = aapq.i / 
					    r__1;
				    ompq.r = q__1.r, ompq.i = q__1.i;
				    notrot = 0;
/* [RTD]      ROTATED  = ROTATED + 1 */
				    pskipped = 0;
				    ++iswrot;

				    if (rotok) {

					aqoap = aaqq / aapp;
					apoaq = aapp / aaqq;
					theta = (r__1 = aqoap - apoaq, abs(
						r__1)) * -.5f / aapq1;
					if (aaqq > aapp0) {
					    theta = -theta;
					}

					if (abs(theta) > bigtheta) {
					    t = .5f / theta;
					    cs = 1.f;
					    r_cnjg(&q__2, &ompq);
					    q__1.r = t * q__2.r, q__1.i = t * 
						    q__2.i;
					    crot_(m, &a[p * a_dim1 + 1], &
						    c__1, &a[q * a_dim1 + 1], 
						    &c__1, &cs, &q__1);
					    if (rsvec) {
			  r_cnjg(&q__2, &ompq);
			  q__1.r = t * q__2.r, q__1.i = t * q__2.i;
			  crot_(&mvl, &v[p * v_dim1 + 1], &c__1, &v[q * 
				  v_dim1 + 1], &c__1, &cs, &q__1);
					    }
/* Computing MAX */
					    r__1 = 0.f, r__2 = t * apoaq * 
						    aapq1 + 1.f;
					    sva[q] = aaqq * M(sqrt)((f2cmax(r__1,
						    r__2)));
/* Computing MAX */
					    r__1 = 0.f, r__2 = 1.f - t * 
						    aqoap * aapq1;
					    aapp *= M(sqrt)((f2cmax(r__1,r__2)));
/* Computing MAX */
					    r__1 = mxsinj, r__2 = abs(t);
					    mxsinj = f2cmax(r__1,r__2);
					} else {


					    thsign = -r_sign(&c_b27, &aapq1);
					    if (aaqq > aapp0) {
			  thsign = -thsign;
					    }
					    t = 1.f / (theta + thsign * M(sqrt)(
						    theta * theta + 1.f));
					    cs = M(sqrt)(1.f / (t * t + 1.f));
					    sn = t * cs;
/* Computing MAX */
					    r__1 = mxsinj, r__2 = abs(sn);
					    mxsinj = f2cmax(r__1,r__2);
/* Computing MAX */
					    r__1 = 0.f, r__2 = t * apoaq * 
						    aapq1 + 1.f;
					    sva[q] = aaqq * M(sqrt)((f2cmax(r__1,
						    r__2)));
/* Computing MAX */
					    r__1 = 0.f, r__2 = 1.f - t * 
						    aqoap * aapq1;
					    aapp *= M(sqrt)((f2cmax(r__1,r__2)));

					    r_cnjg(&q__2, &ompq);
					    q__1.r = sn * q__2.r, q__1.i = sn 
						    * q__2.i;
					    crot_(m, &a[p * a_dim1 + 1], &
						    c__1, &a[q * a_dim1 + 1], 
						    &c__1, &cs, &q__1);
					    if (rsvec) {
			  r_cnjg(&q__2, &ompq);
			  q__1.r = sn * q__2.r, q__1.i = sn * q__2.i;
			  crot_(&mvl, &v[p * v_dim1 + 1], &c__1, &v[q * 
				  v_dim1 + 1], &c__1, &cs, &q__1);
					    }
					}
					i__6 = p;
					i__7 = q;
					q__2.r = -d__[i__7].r, q__2.i = -d__[
						i__7].i;
					q__1.r = q__2.r * ompq.r - q__2.i * 
						ompq.i, q__1.i = q__2.r * 
						ompq.i + q__2.i * ompq.r;
					d__[i__6].r = q__1.r, d__[i__6].i = 
						q__1.i;

				    } else {
					if (aapp > aaqq) {
					    ccopy_(m, &a[p * a_dim1 + 1], &
						    c__1, &work[1], &c__1);
					    clascl_("G", &c__0, &c__0, &aapp, 
						    &c_b27, m, &c__1, &work[1]
						    , lda, &ierr);
					    clascl_("G", &c__0, &c__0, &aaqq, 
						    &c_b27, m, &c__1, &a[q * 
						    a_dim1 + 1], lda, &ierr);
					    q__1.r = -aapq.r, q__1.i = 
						    -aapq.i;
					    caxpy_(m, &q__1, &work[1], &c__1, 
						    &a[q * a_dim1 + 1], &c__1)
						    ;
					    clascl_("G", &c__0, &c__0, &c_b27,
						     &aaqq, m, &c__1, &a[q * 
						    a_dim1 + 1], lda, &ierr);
/* Computing MAX */
					    r__1 = 0.f, r__2 = 1.f - aapq1 * 
						    aapq1;
					    sva[q] = aaqq * M(sqrt)((f2cmax(r__1,
						    r__2)));
					    mxsinj = f2cmax(mxsinj,*sfmin);
					} else {
					    ccopy_(m, &a[q * a_dim1 + 1], &
						    c__1, &work[1], &c__1);
					    clascl_("G", &c__0, &c__0, &aaqq, 
						    &c_b27, m, &c__1, &work[1]
						    , lda, &ierr);
					    clascl_("G", &c__0, &c__0, &aapp, 
						    &c_b27, m, &c__1, &a[p * 
						    a_dim1 + 1], lda, &ierr);
					    r_cnjg(&q__2, &aapq);
					    q__1.r = -q__2.r, q__1.i = 
						    -q__2.i;
					    caxpy_(m, &q__1, &work[1], &c__1, 
						    &a[p * a_dim1 + 1], &c__1)
						    ;
					    clascl_("G", &c__0, &c__0, &c_b27,
						     &aapp, m, &c__1, &a[p * 
						    a_dim1 + 1], lda, &ierr);
/* Computing MAX */
					    r__1 = 0.f, r__2 = 1.f - aapq1 * 
						    aapq1;
					    sva[p] = aapp * M(sqrt)((f2cmax(r__1,
						    r__2)));
					    mxsinj = f2cmax(mxsinj,*sfmin);
					}
				    }
/*           END IF ROTOK THEN ... ELSE */

/*           In the case of cancellation in updating SVA(q), SVA(p) */
/* Computing 2nd power */
				    r__1 = sva[q] / aaqq;
				    if (r__1 * r__1 <= rooteps) {
					if (aaqq < rootbig && aaqq > 
						rootsfmin) {
					    sva[q] = scnrm2_(m, &a[q * a_dim1 
						    + 1], &c__1);
					} else {
					    t = 0.f;
					    aaqq = 1.f;
					    classq_(m, &a[q * a_dim1 + 1], &
						    c__1, &t, &aaqq);
					    sva[q] = t * M(sqrt)(aaqq);
					}
				    }
/* Computing 2nd power */
				    r__1 = aapp / aapp0;
				    if (r__1 * r__1 <= rooteps) {
					if (aapp < rootbig && aapp > 
						rootsfmin) {
					    aapp = scnrm2_(m, &a[p * a_dim1 + 
						    1], &c__1);
					} else {
					    t = 0.f;
					    aapp = 1.f;
					    classq_(m, &a[p * a_dim1 + 1], &
						    c__1, &t, &aapp);
					    aapp = t * M(sqrt)(aapp);
					}
					sva[p] = aapp;
				    }
/*              end of OK rotation */
				} else {
				    ++notrot;
/* [RTD]      SKIPPED  = SKIPPED  + 1 */
				    ++pskipped;
				    ++ijblsk;
				}
			    } else {
				++notrot;
				++pskipped;
				++ijblsk;
			    }

			    if (i__ <= swband && ijblsk >= blskip) {
				sva[p] = aapp;
				notrot = 0;
				goto L2011;
			    }
			    if (i__ <= swband && pskipped > rowskip) {
				aapp = -aapp;
				notrot = 0;
				goto L2203;
			    }

/* L2200: */
			}
/*        end of the q-loop */
L2203:

			sva[p] = aapp;

		    } else {

			if (aapp == 0.f) {
/* Computing MIN */
			    i__5 = jgl + kbl - 1;
			    notrot = notrot + f2cmin(i__5,*n) - jgl + 1;
			}
			if (aapp < 0.f) {
			    notrot = 0;
			}

		    }

/* L2100: */
		}
/*     end of the p-loop */
/* L2010: */
	    }
/*     end of the jbc-loop */
L2011:
/* 2011 bailed out of the jbc-loop */
/* Computing MIN */
	    i__4 = igl + kbl - 1;
	    i__3 = f2cmin(i__4,*n);
	    for (p = igl; p <= i__3; ++p) {
		sva[p] = (r__1 = sva[p], abs(r__1));
/* L2012: */
	    }
/* ** */
/* L2000: */
	}
/* 2000  end of the ibr-loop */

	if (sva[*n] < rootbig && sva[*n] > rootsfmin) {
	    sva[*n] = scnrm2_(m, &a[*n * a_dim1 + 1], &c__1);
	} else {
	    t = 0.f;
	    aapp = 1.f;
	    classq_(m, &a[*n * a_dim1 + 1], &c__1, &t, &aapp);
	    sva[*n] = t * M(sqrt)(aapp);
	}

/*     Additional steering devices */

	if (i__ < swband && (mxaapq <= roottol || iswrot <= *n)) {
	    swband = i__;
	}

	if (i__ > swband + 1 && mxaapq < M(sqrt)((real) (*n)) * *tol && (real) (*
		n) * mxaapq * mxsinj < *tol) {
	    goto L1994;
	}

	if (notrot >= emptsw) {
	    goto L1994;
	}

/* L1993: */
    }
/*     end i=1:NSWEEP loop */

/* #:( Reaching this point means that the procedure has not converged. */
    *info = *nsweep - 1;
    goto L1995;

L1994:
/* #:) Reaching this point means numerical convergence after the i-th */
/*     sweep. */

    *info = 0;
/* #:) INFO = 0 confirms successful iterations. */
L1995:

/*     Sort the vector SVA() of column norms. */
    i__1 = *n - 1;
    for (p = 1; p <= i__1; ++p) {
	i__2 = *n - p + 1;
	q = isamax_(&i__2, &sva[p], &c__1) + p - 1;
	if (p != q) {
	    temp1 = sva[p];
	    sva[p] = sva[q];
	    sva[q] = temp1;
	    i__2 = p;
	    aapq.r = d__[i__2].r, aapq.i = d__[i__2].i;
	    i__2 = p;
	    i__3 = q;
	    d__[i__2].r = d__[i__3].r, d__[i__2].i = d__[i__3].i;
	    i__2 = q;
	    d__[i__2].r = aapq.r, d__[i__2].i = aapq.i;
	    cswap_(m, &a[p * a_dim1 + 1], &c__1, &a[q * a_dim1 + 1], &c__1);
	    if (rsvec) {
		cswap_(&mvl, &v[p * v_dim1 + 1], &c__1, &v[q * v_dim1 + 1], &
			c__1);
	    }
	}
/* L5991: */
    }

    return;
} /* cgsvj0_ */

