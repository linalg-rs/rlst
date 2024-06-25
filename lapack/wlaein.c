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

/* > \brief \b ZLAEIN computes a specified right or left eigenvector of an upper Hessenberg matrix by inverse 
iteration. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLAEIN + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlaein.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlaein.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlaein.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLAEIN( RIGHTV, NOINIT, N, H, LDH, W, V, B, LDB, RWORK, */
/*                          EPS3, SMLNUM, INFO ) */

/*       LOGICAL            NOINIT, RIGHTV */
/*       INTEGER            INFO, LDB, LDH, N */
/*       DOUBLE PRECISION   EPS3, SMLNUM */
/*       COMPLEX*16         W */
/*       DOUBLE PRECISION   RWORK( * ) */
/*       COMPLEX*16         B( LDB, * ), H( LDH, * ), V( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLAEIN uses inverse iteration to find a right or left eigenvector */
/* > corresponding to the eigenvalue W of a complex upper Hessenberg */
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
/* >          = .TRUE. : no initial vector supplied in V */
/* >          = .FALSE.: initial vector supplied in V. */
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
/* >          H is COMPLEX*16 array, dimension (LDH,N) */
/* >          The upper Hessenberg matrix H. */
/* > \endverbatim */
/* > */
/* > \param[in] LDH */
/* > \verbatim */
/* >          LDH is INTEGER */
/* >          The leading dimension of the array H.  LDH >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] W */
/* > \verbatim */
/* >          W is COMPLEX*16 */
/* >          The eigenvalue of H whose corresponding right or left */
/* >          eigenvector is to be computed. */
/* > \endverbatim */
/* > */
/* > \param[in,out] V */
/* > \verbatim */
/* >          V is COMPLEX*16 array, dimension (N) */
/* >          On entry, if NOINIT = .FALSE., V must contain a starting */
/* >          vector for inverse iteration; otherwise V need not be set. */
/* >          On exit, V contains the computed eigenvector, normalized so */
/* >          that the component of largest magnitude has magnitude 1; here */
/* >          the magnitude of a complex number (x,y) is taken to be */
/* >          |x| + |y|. */
/* > \endverbatim */
/* > */
/* > \param[out] B */
/* > \verbatim */
/* >          B is COMPLEX*16 array, dimension (LDB,N) */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B.  LDB >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is DOUBLE PRECISION array, dimension (N) */
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
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          = 1:  inverse iteration did not converge; V is set to the */
/* >                last iterate. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16OTHERauxiliary */

/*  ===================================================================== */
void  wlaein_(logical *rightv, logical *noinit, integer *n, 
	quadcomplex *h__, integer *ldh, quadcomplex *w, quadcomplex *v, 
	quadcomplex *b, integer *ldb, quadreal *rwork, quadreal *eps3, 
	quadreal *smlnum, integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, h_dim1, h_offset, i__1, i__2, i__3, i__4, i__5;
    quadreal d__1, d__2, d__3, d__4;
    quadcomplex z__1, z__2;

    /* Local variables */
    integer i__, j;
    quadcomplex x, ei, ej;
    integer its, ierr;
    quadcomplex temp;
    quadreal scale;
    char trans[1];
    quadreal rtemp, rootn, vnorm;
    extern quadreal qwnrm2_(integer *, quadcomplex *, integer *);
    extern void  wqscal_(integer *, quadreal *, 
	    quadcomplex *, integer *);
    extern integer iwamax_(integer *, quadcomplex *, integer *);
    extern /* Double Complex */ VOID wladiv_(quadcomplex *, quadcomplex *,
	     quadcomplex *);
    char normin[1];
    extern quadreal qwasum_(integer *, quadcomplex *, integer *);
    quadreal nrmsml;
    extern void  wlatrs_(char *, char *, char *, char *, 
	    integer *, quadcomplex *, integer *, quadcomplex *, 
	    quadreal *, quadreal *, integer *);
    quadreal growto;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --v;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --rwork;

    /* Function Body */
    *info = 0;

/*     GROWTO is the threshold used in the acceptance test for an */
/*     eigenvector. */

    rootn = M(sqrt)((quadreal) (*n));
    growto = .1 / rootn;
/* Computing MAX */
    d__1 = 1., d__2 = *eps3 * rootn;
    nrmsml = f2cmax(d__1,d__2) * *smlnum;

/*     Form B = H - W*I (except that the subdiagonal elements are not */
/*     stored). */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * b_dim1;
	    i__4 = i__ + j * h_dim1;
	    b[i__3].r = h__[i__4].r, b[i__3].i = h__[i__4].i;
/* L10: */
	}
	i__2 = j + j * b_dim1;
	i__3 = j + j * h_dim1;
	z__1.r = h__[i__3].r - w->r, z__1.i = h__[i__3].i - w->i;
	b[i__2].r = z__1.r, b[i__2].i = z__1.i;
/* L20: */
    }

    if (*noinit) {

/*        Initialize V. */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__;
	    v[i__2].r = *eps3, v[i__2].i = 0.;
/* L30: */
	}
    } else {

/*        Scale supplied initial vector. */

	vnorm = qwnrm2_(n, &v[1], &c__1);
	d__1 = *eps3 * rootn / f2cmax(vnorm,nrmsml);
	wqscal_(n, &d__1, &v[1], &c__1);
    }

    if (*rightv) {

/*        LU decomposition with partial pivoting of B, replacing zero */
/*        pivots by EPS3. */

	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__ + 1 + i__ * h_dim1;
	    ei.r = h__[i__2].r, ei.i = h__[i__2].i;
	    i__2 = i__ + i__ * b_dim1;
	    if ((d__1 = b[i__2].r, abs(d__1)) + (d__2 = d_imag(&b[i__ + i__ * 
		    b_dim1]), abs(d__2)) < (d__3 = ei.r, abs(d__3)) + (d__4 = 
		    d_imag(&ei), abs(d__4))) {

/*              Interchange rows and eliminate. */

		wladiv_(&z__1, &b[i__ + i__ * b_dim1], &ei);
		x.r = z__1.r, x.i = z__1.i;
		i__2 = i__ + i__ * b_dim1;
		b[i__2].r = ei.r, b[i__2].i = ei.i;
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    i__3 = i__ + 1 + j * b_dim1;
		    temp.r = b[i__3].r, temp.i = b[i__3].i;
		    i__3 = i__ + 1 + j * b_dim1;
		    i__4 = i__ + j * b_dim1;
		    z__2.r = x.r * temp.r - x.i * temp.i, z__2.i = x.r * 
			    temp.i + x.i * temp.r;
		    z__1.r = b[i__4].r - z__2.r, z__1.i = b[i__4].i - z__2.i;
		    b[i__3].r = z__1.r, b[i__3].i = z__1.i;
		    i__3 = i__ + j * b_dim1;
		    b[i__3].r = temp.r, b[i__3].i = temp.i;
/* L40: */
		}
	    } else {

/*              Eliminate without interchange. */

		i__2 = i__ + i__ * b_dim1;
		if (b[i__2].r == 0. && b[i__2].i == 0.) {
		    i__3 = i__ + i__ * b_dim1;
		    b[i__3].r = *eps3, b[i__3].i = 0.;
		}
		wladiv_(&z__1, &ei, &b[i__ + i__ * b_dim1]);
		x.r = z__1.r, x.i = z__1.i;
		if (x.r != 0. || x.i != 0.) {
		    i__2 = *n;
		    for (j = i__ + 1; j <= i__2; ++j) {
			i__3 = i__ + 1 + j * b_dim1;
			i__4 = i__ + 1 + j * b_dim1;
			i__5 = i__ + j * b_dim1;
			z__2.r = x.r * b[i__5].r - x.i * b[i__5].i, z__2.i = 
				x.r * b[i__5].i + x.i * b[i__5].r;
			z__1.r = b[i__4].r - z__2.r, z__1.i = b[i__4].i - 
				z__2.i;
			b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L50: */
		    }
		}
	    }
/* L60: */
	}
	i__1 = *n + *n * b_dim1;
	if (b[i__1].r == 0. && b[i__1].i == 0.) {
	    i__2 = *n + *n * b_dim1;
	    b[i__2].r = *eps3, b[i__2].i = 0.;
	}

	*(unsigned char *)trans = 'N';

    } else {

/*        UL decomposition with partial pivoting of B, replacing zero */
/*        pivots by EPS3. */

	for (j = *n; j >= 2; --j) {
	    i__1 = j + (j - 1) * h_dim1;
	    ej.r = h__[i__1].r, ej.i = h__[i__1].i;
	    i__1 = j + j * b_dim1;
	    if ((d__1 = b[i__1].r, abs(d__1)) + (d__2 = d_imag(&b[j + j * 
		    b_dim1]), abs(d__2)) < (d__3 = ej.r, abs(d__3)) + (d__4 = 
		    d_imag(&ej), abs(d__4))) {

/*              Interchange columns and eliminate. */

		wladiv_(&z__1, &b[j + j * b_dim1], &ej);
		x.r = z__1.r, x.i = z__1.i;
		i__1 = j + j * b_dim1;
		b[i__1].r = ej.r, b[i__1].i = ej.i;
		i__1 = j - 1;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__ + (j - 1) * b_dim1;
		    temp.r = b[i__2].r, temp.i = b[i__2].i;
		    i__2 = i__ + (j - 1) * b_dim1;
		    i__3 = i__ + j * b_dim1;
		    z__2.r = x.r * temp.r - x.i * temp.i, z__2.i = x.r * 
			    temp.i + x.i * temp.r;
		    z__1.r = b[i__3].r - z__2.r, z__1.i = b[i__3].i - z__2.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = i__ + j * b_dim1;
		    b[i__2].r = temp.r, b[i__2].i = temp.i;
/* L70: */
		}
	    } else {

/*              Eliminate without interchange. */

		i__1 = j + j * b_dim1;
		if (b[i__1].r == 0. && b[i__1].i == 0.) {
		    i__2 = j + j * b_dim1;
		    b[i__2].r = *eps3, b[i__2].i = 0.;
		}
		wladiv_(&z__1, &ej, &b[j + j * b_dim1]);
		x.r = z__1.r, x.i = z__1.i;
		if (x.r != 0. || x.i != 0.) {
		    i__1 = j - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			i__2 = i__ + (j - 1) * b_dim1;
			i__3 = i__ + (j - 1) * b_dim1;
			i__4 = i__ + j * b_dim1;
			z__2.r = x.r * b[i__4].r - x.i * b[i__4].i, z__2.i = 
				x.r * b[i__4].i + x.i * b[i__4].r;
			z__1.r = b[i__3].r - z__2.r, z__1.i = b[i__3].i - 
				z__2.i;
			b[i__2].r = z__1.r, b[i__2].i = z__1.i;
/* L80: */
		    }
		}
	    }
/* L90: */
	}
	i__1 = b_dim1 + 1;
	if (b[i__1].r == 0. && b[i__1].i == 0.) {
	    i__2 = b_dim1 + 1;
	    b[i__2].r = *eps3, b[i__2].i = 0.;
	}

	*(unsigned char *)trans = 'C';

    }

    *(unsigned char *)normin = 'N';
    i__1 = *n;
    for (its = 1; its <= i__1; ++its) {

/*        Solve U*x = scale*v for a right eigenvector */
/*          or U**H *x = scale*v for a left eigenvector, */
/*        overwriting x on v. */

	wlatrs_("Upper", trans, "Nonunit", normin, n, &b[b_offset], ldb, &v[1]
		, &scale, &rwork[1], &ierr);
	*(unsigned char *)normin = 'Y';

/*        Test for sufficient growth in the norm of v. */

	vnorm = qwasum_(n, &v[1], &c__1);
	if (vnorm >= growto * scale) {
	    goto L120;
	}

/*        Choose new orthogonal starting vector and try again. */

	rtemp = *eps3 / (rootn + 1.);
	v[1].r = *eps3, v[1].i = 0.;
	i__2 = *n;
	for (i__ = 2; i__ <= i__2; ++i__) {
	    i__3 = i__;
	    v[i__3].r = rtemp, v[i__3].i = 0.;
/* L100: */
	}
	i__2 = *n - its + 1;
	i__3 = *n - its + 1;
	d__1 = *eps3 * rootn;
	z__1.r = v[i__3].r - d__1, z__1.i = v[i__3].i;
	v[i__2].r = z__1.r, v[i__2].i = z__1.i;
/* L110: */
    }

/*     Failure to find eigenvector in N iterations. */

    *info = 1;

L120:

/*     Normalize eigenvector. */

    i__ = iwamax_(n, &v[1], &c__1);
    i__1 = i__;
    d__3 = 1. / ((d__1 = v[i__1].r, abs(d__1)) + (d__2 = d_imag(&v[i__]), abs(
	    d__2)));
    wqscal_(n, &d__3, &v[1], &c__1);

    return;

/*     End of ZLAEIN */

} /* wlaein_ */

