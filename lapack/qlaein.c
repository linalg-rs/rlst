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

/* > \brief \b DLAEIN computes a specified right or left eigenvector of an upper Hessenberg matrix by inverse 
iteration. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DLAEIN + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaein.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaein.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaein.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DLAEIN( RIGHTV, NOINIT, N, H, LDH, WR, WI, VR, VI, B, */
/*                          LDB, WORK, EPS3, SMLNUM, BIGNUM, INFO ) */

/*       LOGICAL            NOINIT, RIGHTV */
/*       INTEGER            INFO, LDB, LDH, N */
/*       DOUBLE PRECISION   BIGNUM, EPS3, SMLNUM, WI, WR */
/*       DOUBLE PRECISION   B( LDB, * ), H( LDH, * ), VI( * ), VR( * ), */
/*      $                   WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DLAEIN uses inverse iteration to find a right or left eigenvector */
/* > corresponding to the eigenvalue (WR,WI) of a doublereal upper Hessenberg */
/* > matrix H. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] RIGHTV */
/* > \verbatim */
/* >          RIGHTV is LOGICAL */
/* >          = .TRUE. : compute right eigenvector; */
/* >          = .FALSE.: compute left eigenvector. */
/* > \endverbatim */
/* > */
/* > \param[in] NOINIT */
/* > \verbatim */
/* >          NOINIT is LOGICAL */
/* >          = .TRUE. : no initial vector supplied in (VR,VI). */
/* >          = .FALSE.: initial vector supplied in (VR,VI). */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix H.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] H */
/* > \verbatim */
/* >          H is DOUBLE PRECISION array, dimension (LDH,N) */
/* >          The upper Hessenberg matrix H. */
/* > \endverbatim */
/* > */
/* > \param[in] LDH */
/* > \verbatim */
/* >          LDH is INTEGER */
/* >          The leading dimension of the array H.  LDH >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] WR */
/* > \verbatim */
/* >          WR is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[in] WI */
/* > \verbatim */
/* >          WI is DOUBLE PRECISION */
/* >          The doublereal and imaginary parts of the eigenvalue of H whose */
/* >          corresponding right or left eigenvector is to be computed. */
/* > \endverbatim */
/* > */
/* > \param[in,out] VR */
/* > \verbatim */
/* >          VR is DOUBLE PRECISION array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[in,out] VI */
/* > \verbatim */
/* >          VI is DOUBLE PRECISION array, dimension (N) */
/* >          On entry, if NOINIT = .FALSE. and WI = 0.0, VR must contain */
/* >          a doublereal starting vector for inverse iteration using the doublereal */
/* >          eigenvalue WR; if NOINIT = .FALSE. and WI.ne.0.0, VR and VI */
/* >          must contain the doublereal and imaginary parts of a complex */
/* >          starting vector for inverse iteration using the complex */
/* >          eigenvalue (WR,WI); otherwise VR and VI need not be set. */
/* >          On exit, if WI = 0.0 (doublereal eigenvalue), VR contains the */
/* >          computed doublereal eigenvector; if WI.ne.0.0 (complex eigenvalue), */
/* >          VR and VI contain the doublereal and imaginary parts of the */
/* >          computed complex eigenvector. The eigenvector is normalized */
/* >          so that the component of largest magnitude has magnitude 1; */
/* >          here the magnitude of a complex number (x,y) is taken to be */
/* >          |x| + |y|. */
/* >          VI is not referenced if WI = 0.0. */
/* > \endverbatim */
/* > */
/* > \param[out] B */
/* > \verbatim */
/* >          B is DOUBLE PRECISION array, dimension (LDB,N) */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B.  LDB >= N+1. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[in] EPS3 */
/* > \verbatim */
/* >          EPS3 is DOUBLE PRECISION */
/* >          A small machine-dependent value which is used to perturb */
/* >          close eigenvalues, and to replace zero pivots. */
/* > \endverbatim */
/* > */
/* > \param[in] SMLNUM */
/* > \verbatim */
/* >          SMLNUM is DOUBLE PRECISION */
/* >          A machine-dependent value close to the underflow threshold. */
/* > \endverbatim */
/* > */
/* > \param[in] BIGNUM */
/* > \verbatim */
/* >          BIGNUM is DOUBLE PRECISION */
/* >          A machine-dependent value close to the overflow threshold. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          = 1:  inverse iteration did not converge; VR is set to the */
/* >                last iterate, and so is VI if WI.ne.0.0. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup doubleOTHERauxiliary */

/*  ===================================================================== */
void  qlaein_(logical *rightv, logical *noinit, integer *n, 
	quadreal *h__, integer *ldh, quadreal *wr, quadreal *wi, 
	quadreal *vr, quadreal *vi, quadreal *b, integer *ldb, 
	quadreal *work, quadreal *eps3, quadreal *smlnum, quadreal *
	bignum, integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, h_dim1, h_offset, i__1, i__2, i__3, i__4;
    quadreal d__1, d__2, d__3, d__4;

    /* Local variables */
    integer i__, j;
    quadreal w, x, y;
    integer i1, i2, i3;
    quadreal w1, ei, ej, xi, xr, rec;
    integer its, ierr;
    quadreal temp, norm, vmax;
    extern quadreal qnrm2_(integer *, quadreal *, integer *);
    extern void  qscal_(integer *, quadreal *, quadreal *, 
	    integer *);
    quadreal scale;
    extern quadreal qasum_(integer *, quadreal *, integer *);
    char trans[1];
    quadreal vcrit, rootn, vnorm;
    extern quadreal qlapy2_(quadreal *, quadreal *);
    quadreal absbii, absbjj;
    extern integer iqamax_(integer *, quadreal *, integer *);
    extern void  qladiv_(quadreal *, quadreal *, 
	    quadreal *, quadreal *, quadreal *, quadreal *), qlatrs_(
	    char *, char *, char *, char *, integer *, quadreal *, integer *
	    , quadreal *, quadreal *, quadreal *, integer *);
    char normin[1];
    quadreal nrmsml, growto;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --vr;
    --vi;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --work;

    /* Function Body */
    *info = 0;

/*     GROWTO is the threshold used in the acceptance test for an */
/*     eigenvector. */

    rootn = M(sqrt)((quadreal) (*n));
    growto = .1 / rootn;
/* Computing MAX */
    d__1 = 1., d__2 = *eps3 * rootn;
    nrmsml = f2cmax(d__1,d__2) * *smlnum;

/*     Form B = H - (WR,WI)*I (except that the subdiagonal elements and */
/*     the imaginary parts of the diagonal elements are not stored). */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    b[i__ + j * b_dim1] = h__[i__ + j * h_dim1];
/* L10: */
	}
	b[j + j * b_dim1] = h__[j + j * h_dim1] - *wr;
/* L20: */
    }

    if (*wi == 0.) {

/*        Real eigenvalue. */

	if (*noinit) {

/*           Set initial vector. */

	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		vr[i__] = *eps3;
/* L30: */
	    }
	} else {

/*           Scale supplied initial vector. */

	    vnorm = qnrm2_(n, &vr[1], &c__1);
	    d__1 = *eps3 * rootn / f2cmax(vnorm,nrmsml);
	    qscal_(n, &d__1, &vr[1], &c__1);
	}

	if (*rightv) {

/*           LU decomposition with partial pivoting of B, replacing zero */
/*           pivots by EPS3. */

	    i__1 = *n - 1;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		ei = h__[i__ + 1 + i__ * h_dim1];
		if ((d__1 = b[i__ + i__ * b_dim1], abs(d__1)) < abs(ei)) {

/*                 Interchange rows and eliminate. */

		    x = b[i__ + i__ * b_dim1] / ei;
		    b[i__ + i__ * b_dim1] = ei;
		    i__2 = *n;
		    for (j = i__ + 1; j <= i__2; ++j) {
			temp = b[i__ + 1 + j * b_dim1];
			b[i__ + 1 + j * b_dim1] = b[i__ + j * b_dim1] - x * 
				temp;
			b[i__ + j * b_dim1] = temp;
/* L40: */
		    }
		} else {

/*                 Eliminate without interchange. */

		    if (b[i__ + i__ * b_dim1] == 0.) {
			b[i__ + i__ * b_dim1] = *eps3;
		    }
		    x = ei / b[i__ + i__ * b_dim1];
		    if (x != 0.) {
			i__2 = *n;
			for (j = i__ + 1; j <= i__2; ++j) {
			    b[i__ + 1 + j * b_dim1] -= x * b[i__ + j * b_dim1]
				    ;
/* L50: */
			}
		    }
		}
/* L60: */
	    }
	    if (b[*n + *n * b_dim1] == 0.) {
		b[*n + *n * b_dim1] = *eps3;
	    }

	    *(unsigned char *)trans = 'N';

	} else {

/*           UL decomposition with partial pivoting of B, replacing zero */
/*           pivots by EPS3. */

	    for (j = *n; j >= 2; --j) {
		ej = h__[j + (j - 1) * h_dim1];
		if ((d__1 = b[j + j * b_dim1], abs(d__1)) < abs(ej)) {

/*                 Interchange columns and eliminate. */

		    x = b[j + j * b_dim1] / ej;
		    b[j + j * b_dim1] = ej;
		    i__1 = j - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			temp = b[i__ + (j - 1) * b_dim1];
			b[i__ + (j - 1) * b_dim1] = b[i__ + j * b_dim1] - x * 
				temp;
			b[i__ + j * b_dim1] = temp;
/* L70: */
		    }
		} else {

/*                 Eliminate without interchange. */

		    if (b[j + j * b_dim1] == 0.) {
			b[j + j * b_dim1] = *eps3;
		    }
		    x = ej / b[j + j * b_dim1];
		    if (x != 0.) {
			i__1 = j - 1;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    b[i__ + (j - 1) * b_dim1] -= x * b[i__ + j * 
				    b_dim1];
/* L80: */
			}
		    }
		}
/* L90: */
	    }
	    if (b[b_dim1 + 1] == 0.) {
		b[b_dim1 + 1] = *eps3;
	    }

	    *(unsigned char *)trans = 'T';

	}

	*(unsigned char *)normin = 'N';
	i__1 = *n;
	for (its = 1; its <= i__1; ++its) {

/*           Solve U*x = scale*v for a right eigenvector */
/*             or U**T*x = scale*v for a left eigenvector, */
/*           overwriting x on v. */

	    qlatrs_("Upper", trans, "Nonunit", normin, n, &b[b_offset], ldb, &
		    vr[1], &scale, &work[1], &ierr);
	    *(unsigned char *)normin = 'Y';

/*           Test for sufficient growth in the norm of v. */

	    vnorm = qasum_(n, &vr[1], &c__1);
	    if (vnorm >= growto * scale) {
		goto L120;
	    }

/*           Choose new orthogonal starting vector and try again. */

	    temp = *eps3 / (rootn + 1.);
	    vr[1] = *eps3;
	    i__2 = *n;
	    for (i__ = 2; i__ <= i__2; ++i__) {
		vr[i__] = temp;
/* L100: */
	    }
	    vr[*n - its + 1] -= *eps3 * rootn;
/* L110: */
	}

/*        Failure to find eigenvector in N iterations. */

	*info = 1;

L120:

/*        Normalize eigenvector. */

	i__ = iqamax_(n, &vr[1], &c__1);
	d__2 = 1. / (d__1 = vr[i__], abs(d__1));
	qscal_(n, &d__2, &vr[1], &c__1);
    } else {

/*        Complex eigenvalue. */

	if (*noinit) {

/*           Set initial vector. */

	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		vr[i__] = *eps3;
		vi[i__] = 0.;
/* L130: */
	    }
	} else {

/*           Scale supplied initial vector. */

	    d__1 = qnrm2_(n, &vr[1], &c__1);
	    d__2 = qnrm2_(n, &vi[1], &c__1);
	    norm = qlapy2_(&d__1, &d__2);
	    rec = *eps3 * rootn / f2cmax(norm,nrmsml);
	    qscal_(n, &rec, &vr[1], &c__1);
	    qscal_(n, &rec, &vi[1], &c__1);
	}

	if (*rightv) {

/*           LU decomposition with partial pivoting of B, replacing zero */
/*           pivots by EPS3. */

/*           The imaginary part of the (i,j)-th element of U is stored in */
/*           B(j+1,i). */

	    b[b_dim1 + 2] = -(*wi);
	    i__1 = *n;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		b[i__ + 1 + b_dim1] = 0.;
/* L140: */
	    }

	    i__1 = *n - 1;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		absbii = qlapy2_(&b[i__ + i__ * b_dim1], &b[i__ + 1 + i__ * 
			b_dim1]);
		ei = h__[i__ + 1 + i__ * h_dim1];
		if (absbii < abs(ei)) {

/*                 Interchange rows and eliminate. */

		    xr = b[i__ + i__ * b_dim1] / ei;
		    xi = b[i__ + 1 + i__ * b_dim1] / ei;
		    b[i__ + i__ * b_dim1] = ei;
		    b[i__ + 1 + i__ * b_dim1] = 0.;
		    i__2 = *n;
		    for (j = i__ + 1; j <= i__2; ++j) {
			temp = b[i__ + 1 + j * b_dim1];
			b[i__ + 1 + j * b_dim1] = b[i__ + j * b_dim1] - xr * 
				temp;
			b[j + 1 + (i__ + 1) * b_dim1] = b[j + 1 + i__ * 
				b_dim1] - xi * temp;
			b[i__ + j * b_dim1] = temp;
			b[j + 1 + i__ * b_dim1] = 0.;
/* L150: */
		    }
		    b[i__ + 2 + i__ * b_dim1] = -(*wi);
		    b[i__ + 1 + (i__ + 1) * b_dim1] -= xi * *wi;
		    b[i__ + 2 + (i__ + 1) * b_dim1] += xr * *wi;
		} else {

/*                 Eliminate without interchanging rows. */

		    if (absbii == 0.) {
			b[i__ + i__ * b_dim1] = *eps3;
			b[i__ + 1 + i__ * b_dim1] = 0.;
			absbii = *eps3;
		    }
		    ei = ei / absbii / absbii;
		    xr = b[i__ + i__ * b_dim1] * ei;
		    xi = -b[i__ + 1 + i__ * b_dim1] * ei;
		    i__2 = *n;
		    for (j = i__ + 1; j <= i__2; ++j) {
			b[i__ + 1 + j * b_dim1] = b[i__ + 1 + j * b_dim1] - 
				xr * b[i__ + j * b_dim1] + xi * b[j + 1 + i__ 
				* b_dim1];
			b[j + 1 + (i__ + 1) * b_dim1] = -xr * b[j + 1 + i__ * 
				b_dim1] - xi * b[i__ + j * b_dim1];
/* L160: */
		    }
		    b[i__ + 2 + (i__ + 1) * b_dim1] -= *wi;
		}

/*              Compute 1-norm of offdiagonal elements of i-th row. */

		i__2 = *n - i__;
		i__3 = *n - i__;
		work[i__] = qasum_(&i__2, &b[i__ + (i__ + 1) * b_dim1], ldb) 
			+ qasum_(&i__3, &b[i__ + 2 + i__ * b_dim1], &c__1);
/* L170: */
	    }
	    if (b[*n + *n * b_dim1] == 0. && b[*n + 1 + *n * b_dim1] == 0.) {
		b[*n + *n * b_dim1] = *eps3;
	    }
	    work[*n] = 0.;

	    i1 = *n;
	    i2 = 1;
	    i3 = -1;
	} else {

/*           UL decomposition with partial pivoting of conjg(B), */
/*           replacing zero pivots by EPS3. */

/*           The imaginary part of the (i,j)-th element of U is stored in */
/*           B(j+1,i). */

	    b[*n + 1 + *n * b_dim1] = *wi;
	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		b[*n + 1 + j * b_dim1] = 0.;
/* L180: */
	    }

	    for (j = *n; j >= 2; --j) {
		ej = h__[j + (j - 1) * h_dim1];
		absbjj = qlapy2_(&b[j + j * b_dim1], &b[j + 1 + j * b_dim1]);
		if (absbjj < abs(ej)) {

/*                 Interchange columns and eliminate */

		    xr = b[j + j * b_dim1] / ej;
		    xi = b[j + 1 + j * b_dim1] / ej;
		    b[j + j * b_dim1] = ej;
		    b[j + 1 + j * b_dim1] = 0.;
		    i__1 = j - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			temp = b[i__ + (j - 1) * b_dim1];
			b[i__ + (j - 1) * b_dim1] = b[i__ + j * b_dim1] - xr *
				 temp;
			b[j + i__ * b_dim1] = b[j + 1 + i__ * b_dim1] - xi * 
				temp;
			b[i__ + j * b_dim1] = temp;
			b[j + 1 + i__ * b_dim1] = 0.;
/* L190: */
		    }
		    b[j + 1 + (j - 1) * b_dim1] = *wi;
		    b[j - 1 + (j - 1) * b_dim1] += xi * *wi;
		    b[j + (j - 1) * b_dim1] -= xr * *wi;
		} else {

/*                 Eliminate without interchange. */

		    if (absbjj == 0.) {
			b[j + j * b_dim1] = *eps3;
			b[j + 1 + j * b_dim1] = 0.;
			absbjj = *eps3;
		    }
		    ej = ej / absbjj / absbjj;
		    xr = b[j + j * b_dim1] * ej;
		    xi = -b[j + 1 + j * b_dim1] * ej;
		    i__1 = j - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			b[i__ + (j - 1) * b_dim1] = b[i__ + (j - 1) * b_dim1] 
				- xr * b[i__ + j * b_dim1] + xi * b[j + 1 + 
				i__ * b_dim1];
			b[j + i__ * b_dim1] = -xr * b[j + 1 + i__ * b_dim1] - 
				xi * b[i__ + j * b_dim1];
/* L200: */
		    }
		    b[j + (j - 1) * b_dim1] += *wi;
		}

/*              Compute 1-norm of offdiagonal elements of j-th column. */

		i__1 = j - 1;
		i__2 = j - 1;
		work[j] = qasum_(&i__1, &b[j * b_dim1 + 1], &c__1) + qasum_(&
			i__2, &b[j + 1 + b_dim1], ldb);
/* L210: */
	    }
	    if (b[b_dim1 + 1] == 0. && b[b_dim1 + 2] == 0.) {
		b[b_dim1 + 1] = *eps3;
	    }
	    work[1] = 0.;

	    i1 = 1;
	    i2 = *n;
	    i3 = 1;
	}

	i__1 = *n;
	for (its = 1; its <= i__1; ++its) {
	    scale = 1.;
	    vmax = 1.;
	    vcrit = *bignum;

/*           Solve U*(xr,xi) = scale*(vr,vi) for a right eigenvector, */
/*             or U**T*(xr,xi) = scale*(vr,vi) for a left eigenvector, */
/*           overwriting (xr,xi) on (vr,vi). */

	    i__2 = i2;
	    i__3 = i3;
	    for (i__ = i1; i__3 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__3) 
		    {

		if (work[i__] > vcrit) {
		    rec = 1. / vmax;
		    qscal_(n, &rec, &vr[1], &c__1);
		    qscal_(n, &rec, &vi[1], &c__1);
		    scale *= rec;
		    vmax = 1.;
		    vcrit = *bignum;
		}

		xr = vr[i__];
		xi = vi[i__];
		if (*rightv) {
		    i__4 = *n;
		    for (j = i__ + 1; j <= i__4; ++j) {
			xr = xr - b[i__ + j * b_dim1] * vr[j] + b[j + 1 + i__ 
				* b_dim1] * vi[j];
			xi = xi - b[i__ + j * b_dim1] * vi[j] - b[j + 1 + i__ 
				* b_dim1] * vr[j];
/* L220: */
		    }
		} else {
		    i__4 = i__ - 1;
		    for (j = 1; j <= i__4; ++j) {
			xr = xr - b[j + i__ * b_dim1] * vr[j] + b[i__ + 1 + j 
				* b_dim1] * vi[j];
			xi = xi - b[j + i__ * b_dim1] * vi[j] - b[i__ + 1 + j 
				* b_dim1] * vr[j];
/* L230: */
		    }
		}

		w = (d__1 = b[i__ + i__ * b_dim1], abs(d__1)) + (d__2 = b[i__ 
			+ 1 + i__ * b_dim1], abs(d__2));
		if (w > *smlnum) {
		    if (w < 1.) {
			w1 = abs(xr) + abs(xi);
			if (w1 > w * *bignum) {
			    rec = 1. / w1;
			    qscal_(n, &rec, &vr[1], &c__1);
			    qscal_(n, &rec, &vi[1], &c__1);
			    xr = vr[i__];
			    xi = vi[i__];
			    scale *= rec;
			    vmax *= rec;
			}
		    }

/*                 Divide by diagonal element of B. */

		    qladiv_(&xr, &xi, &b[i__ + i__ * b_dim1], &b[i__ + 1 + 
			    i__ * b_dim1], &vr[i__], &vi[i__]);
/* Computing MAX */
		    d__3 = (d__1 = vr[i__], abs(d__1)) + (d__2 = vi[i__], abs(
			    d__2));
		    vmax = f2cmax(d__3,vmax);
		    vcrit = *bignum / vmax;
		} else {
		    i__4 = *n;
		    for (j = 1; j <= i__4; ++j) {
			vr[j] = 0.;
			vi[j] = 0.;
/* L240: */
		    }
		    vr[i__] = 1.;
		    vi[i__] = 1.;
		    scale = 0.;
		    vmax = 1.;
		    vcrit = *bignum;
		}
/* L250: */
	    }

/*           Test for sufficient growth in the norm of (VR,VI). */

	    vnorm = qasum_(n, &vr[1], &c__1) + qasum_(n, &vi[1], &c__1);
	    if (vnorm >= growto * scale) {
		goto L280;
	    }

/*           Choose a new orthogonal starting vector and try again. */

	    y = *eps3 / (rootn + 1.);
	    vr[1] = *eps3;
	    vi[1] = 0.;

	    i__3 = *n;
	    for (i__ = 2; i__ <= i__3; ++i__) {
		vr[i__] = y;
		vi[i__] = 0.;
/* L260: */
	    }
	    vr[*n - its + 1] -= *eps3 * rootn;
/* L270: */
	}

/*        Failure to find eigenvector in N iterations */

	*info = 1;

L280:

/*        Normalize eigenvector. */

	vnorm = 0.;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	    d__3 = vnorm, d__4 = (d__1 = vr[i__], abs(d__1)) + (d__2 = vi[i__]
		    , abs(d__2));
	    vnorm = f2cmax(d__3,d__4);
/* L290: */
	}
	d__1 = 1. / vnorm;
	qscal_(n, &d__1, &vr[1], &c__1);
	d__1 = 1. / vnorm;
	qscal_(n, &d__1, &vi[1], &c__1);

    }

    return;

/*     End of DLAEIN */

} /* qlaein_ */

