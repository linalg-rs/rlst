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

static halfcomplex c_b1 = {0.,0.};
static halfcomplex c_b2 = {1.,0.};
static integer c__1 = 1;

/* > \brief \b ZHBGVX */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZHBGVX + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zhbgvx.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zhbgvx.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zhbgvx.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZHBGVX( JOBZ, RANGE, UPLO, N, KA, KB, AB, LDAB, BB, */
/*                          LDBB, Q, LDQ, VL, VU, IL, IU, ABSTOL, M, W, Z, */
/*                          LDZ, WORK, RWORK, IWORK, IFAIL, INFO ) */

/*       CHARACTER          JOBZ, RANGE, UPLO */
/*       INTEGER            IL, INFO, IU, KA, KB, LDAB, LDBB, LDQ, LDZ, M, */
/*      $                   N */
/*       DOUBLE PRECISION   ABSTOL, VL, VU */
/*       INTEGER            IFAIL( * ), IWORK( * ) */
/*       DOUBLE PRECISION   RWORK( * ), W( * ) */
/*       COMPLEX*16         AB( LDAB, * ), BB( LDBB, * ), Q( LDQ, * ), */
/*      $                   WORK( * ), Z( LDZ, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZHBGVX computes all the eigenvalues, and optionally, the eigenvectors */
/* > of a complex generalized Hermitian-definite banded eigenproblem, of */
/* > the form A*x=(lambda)*B*x. Here A and B are assumed to be Hermitian */
/* > and banded, and B is also positive definite.  Eigenvalues and */
/* > eigenvectors can be selected by specifying either all eigenvalues, */
/* > a range of values or a range of indices for the desired eigenvalues. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] JOBZ */
/* > \verbatim */
/* >          JOBZ is CHARACTER*1 */
/* >          = 'N':  Compute eigenvalues only; */
/* >          = 'V':  Compute eigenvalues and eigenvectors. */
/* > \endverbatim */
/* > */
/* > \param[in] RANGE */
/* > \verbatim */
/* >          RANGE is CHARACTER*1 */
/* >          = 'A': all eigenvalues will be found; */
/* >          = 'V': all eigenvalues in the half-open interval (VL,VU] */
/* >                 will be found; */
/* >          = 'I': the IL-th through IU-th eigenvalues will be found. */
/* > \endverbatim */
/* > */
/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          = 'U':  Upper triangles of A and B are stored; */
/* >          = 'L':  Lower triangles of A and B are stored. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrices A and B.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] KA */
/* > \verbatim */
/* >          KA is INTEGER */
/* >          The number of superdiagonals of the matrix A if UPLO = 'U', */
/* >          or the number of subdiagonals if UPLO = 'L'. KA >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] KB */
/* > \verbatim */
/* >          KB is INTEGER */
/* >          The number of superdiagonals of the matrix B if UPLO = 'U', */
/* >          or the number of subdiagonals if UPLO = 'L'. KB >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] AB */
/* > \verbatim */
/* >          AB is COMPLEX*16 array, dimension (LDAB, N) */
/* >          On entry, the upper or lower triangle of the Hermitian band */
/* >          matrix A, stored in the first ka+1 rows of the array.  The */
/* >          j-th column of A is stored in the j-th column of the array AB */
/* >          as follows: */
/* >          if UPLO = 'U', AB(ka+1+i-j,j) = A(i,j) for f2cmax(1,j-ka)<=i<=j; */
/* >          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=f2cmin(n,j+ka). */
/* > */
/* >          On exit, the contents of AB are destroyed. */
/* > \endverbatim */
/* > */
/* > \param[in] LDAB */
/* > \verbatim */
/* >          LDAB is INTEGER */
/* >          The leading dimension of the array AB.  LDAB >= KA+1. */
/* > \endverbatim */
/* > */
/* > \param[in,out] BB */
/* > \verbatim */
/* >          BB is COMPLEX*16 array, dimension (LDBB, N) */
/* >          On entry, the upper or lower triangle of the Hermitian band */
/* >          matrix B, stored in the first kb+1 rows of the array.  The */
/* >          j-th column of B is stored in the j-th column of the array BB */
/* >          as follows: */
/* >          if UPLO = 'U', BB(kb+1+i-j,j) = B(i,j) for f2cmax(1,j-kb)<=i<=j; */
/* >          if UPLO = 'L', BB(1+i-j,j)    = B(i,j) for j<=i<=f2cmin(n,j+kb). */
/* > */
/* >          On exit, the factor S from the split Cholesky factorization */
/* >          B = S**H*S, as returned by ZPBSTF. */
/* > \endverbatim */
/* > */
/* > \param[in] LDBB */
/* > \verbatim */
/* >          LDBB is INTEGER */
/* >          The leading dimension of the array BB.  LDBB >= KB+1. */
/* > \endverbatim */
/* > */
/* > \param[out] Q */
/* > \verbatim */
/* >          Q is COMPLEX*16 array, dimension (LDQ, N) */
/* >          If JOBZ = 'V', the n-by-n matrix used in the reduction of */
/* >          A*x = (lambda)*B*x to standard form, i.e. C*x = (lambda)*x, */
/* >          and consequently C to tridiagonal form. */
/* >          If JOBZ = 'N', the array Q is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDQ */
/* > \verbatim */
/* >          LDQ is INTEGER */
/* >          The leading dimension of the array Q.  If JOBZ = 'N', */
/* >          LDQ >= 1. If JOBZ = 'V', LDQ >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] VL */
/* > \verbatim */
/* >          VL is DOUBLE PRECISION */
/* > */
/* >          If RANGE='V', the lower bound of the interval to */
/* >          be searched for eigenvalues. VL < VU. */
/* >          Not referenced if RANGE = 'A' or 'I'. */
/* > \endverbatim */
/* > */
/* > \param[in] VU */
/* > \verbatim */
/* >          VU is DOUBLE PRECISION */
/* > */
/* >          If RANGE='V', the upper bound of the interval to */
/* >          be searched for eigenvalues. VL < VU. */
/* >          Not referenced if RANGE = 'A' or 'I'. */
/* > \endverbatim */
/* > */
/* > \param[in] IL */
/* > \verbatim */
/* >          IL is INTEGER */
/* > */
/* >          If RANGE='I', the index of the */
/* >          smallest eigenvalue to be returned. */
/* >          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0. */
/* >          Not referenced if RANGE = 'A' or 'V'. */
/* > \endverbatim */
/* > */
/* > \param[in] IU */
/* > \verbatim */
/* >          IU is INTEGER */
/* > */
/* >          If RANGE='I', the index of the */
/* >          largest eigenvalue to be returned. */
/* >          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0. */
/* >          Not referenced if RANGE = 'A' or 'V'. */
/* > \endverbatim */
/* > */
/* > \param[in] ABSTOL */
/* > \verbatim */
/* >          ABSTOL is DOUBLE PRECISION */
/* >          The absolute error tolerance for the eigenvalues. */
/* >          An approximate eigenvalue is accepted as converged */
/* >          when it is determined to lie in an interval [a,b] */
/* >          of width less than or equal to */
/* > */
/* >                  ABSTOL + EPS *   f2cmax( |a|,|b| ) , */
/* > */
/* >          where EPS is the machine precision.  If ABSTOL is less than */
/* >          or equal to zero, then  EPS*|T|  will be used in its place, */
/* >          where |T| is the 1-norm of the tridiagonal matrix obtained */
/* >          by reducing AP to tridiagonal form. */
/* > */
/* >          Eigenvalues will be computed most accurately when ABSTOL is */
/* >          set to twice the underflow threshold 2*DLAMCH('S'), not zero. */
/* >          If this routine returns with INFO>0, indicating that some */
/* >          eigenvectors did not converge, try setting ABSTOL to */
/* >          2*DLAMCH('S'). */
/* > \endverbatim */
/* > */
/* > \param[out] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The total number of eigenvalues found.  0 <= M <= N. */
/* >          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1. */
/* > \endverbatim */
/* > */
/* > \param[out] W */
/* > \verbatim */
/* >          W is DOUBLE PRECISION array, dimension (N) */
/* >          If INFO = 0, the eigenvalues in ascending order. */
/* > \endverbatim */
/* > */
/* > \param[out] Z */
/* > \verbatim */
/* >          Z is COMPLEX*16 array, dimension (LDZ, N) */
/* >          If JOBZ = 'V', then if INFO = 0, Z contains the matrix Z of */
/* >          eigenvectors, with the i-th column of Z holding the */
/* >          eigenvector associated with W(i). The eigenvectors are */
/* >          normalized so that Z**H*B*Z = I. */
/* >          If JOBZ = 'N', then Z is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDZ */
/* > \verbatim */
/* >          LDZ is INTEGER */
/* >          The leading dimension of the array Z.  LDZ >= 1, and if */
/* >          JOBZ = 'V', LDZ >= N. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX*16 array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is DOUBLE PRECISION array, dimension (7*N) */
/* > \endverbatim */
/* > */
/* > \param[out] IWORK */
/* > \verbatim */
/* >          IWORK is INTEGER array, dimension (5*N) */
/* > \endverbatim */
/* > */
/* > \param[out] IFAIL */
/* > \verbatim */
/* >          IFAIL is INTEGER array, dimension (N) */
/* >          If JOBZ = 'V', then if INFO = 0, the first M elements of */
/* >          IFAIL are zero.  If INFO > 0, then IFAIL contains the */
/* >          indices of the eigenvectors that failed to converge. */
/* >          If JOBZ = 'N', then IFAIL is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  if INFO = i, and i is: */
/* >             <= N:  then i eigenvectors failed to converge.  Their */
/* >                    indices are stored in array IFAIL. */
/* >             > N:   if INFO = N + i, for 1 <= i <= N, then ZPBSTF */
/* >                    returned INFO = i: B is not positive definite. */
/* >                    The factorization of B could not be completed and */
/* >                    no eigenvalues or eigenvectors were computed. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2016 */

/* > \ingroup complex16OTHEReigen */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >     Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA */

/*  ===================================================================== */
void  khbgvx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *ka, integer *kb, halfcomplex *ab, integer *ldab, 
	halfcomplex *bb, integer *ldbb, halfcomplex *q, integer *ldq, 
	halfreal *vl, halfreal *vu, integer *il, integer *iu, halfreal *
	abstol, integer *m, halfreal *w, halfcomplex *z__, integer *ldz, 
	halfcomplex *work, halfreal *rwork, integer *iwork, integer *
	ifail, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, bb_dim1, bb_offset, q_dim1, q_offset, z_dim1, 
	    z_offset, i__1, i__2;

    /* Local variables */
    integer i__, j, jj;
    halfreal tmp1;
    integer indd, inde;
    char vect[1];
    logical test;
    integer itmp1, indee;
    extern logical lsame_(char *, char *);
    integer iinfo;
    char order[1];
    extern void  hcopy_(integer *, halfreal *, integer *, 
	    halfreal *, integer *), kgemv_(char *, integer *, integer *, 
	    halfcomplex *, halfcomplex *, integer *, halfcomplex *, 
	    integer *, halfcomplex *, halfcomplex *, integer *);
    logical upper, wantz;
    extern void  kcopy_(integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *), kswap_(integer *, halfcomplex *, 
	    integer *, halfcomplex *, integer *);
    logical alleig, indeig;
    integer indibl;
    logical valeig;
    extern void  xerbla_(char *, integer *);
    integer indiwk, indisp;
    extern void  hsterf_(integer *, halfreal *, halfreal *,
	     integer *), hstebz_(char *, char *, integer *, halfreal *, 
	    halfreal *, integer *, integer *, halfreal *, halfreal *, 
	    halfreal *, integer *, integer *, halfreal *, integer *, 
	    integer *, halfreal *, integer *, integer *), 
	    khbtrd_(char *, char *, integer *, integer *, halfcomplex *, 
	    integer *, halfreal *, halfreal *, halfcomplex *, integer *,
	     halfcomplex *, integer *);
    integer indrwk, indwrk;
    extern void  khbgst_(char *, char *, integer *, integer *, 
	    integer *, halfcomplex *, integer *, halfcomplex *, integer *,
	     halfcomplex *, integer *, halfcomplex *, halfreal *, 
	    integer *), klacpy_(char *, integer *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, integer *);
    integer nsplit;
    extern void  kpbstf_(char *, integer *, integer *, 
	    halfcomplex *, integer *, integer *), kstein_(integer *,
	     halfreal *, halfreal *, integer *, halfreal *, integer *, 
	    integer *, halfcomplex *, integer *, halfreal *, integer *, 
	    integer *, integer *), ksteqr_(char *, integer *, halfreal *, 
	    halfreal *, halfcomplex *, integer *, halfreal *, integer *);


/*  -- LAPACK driver routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    bb_dim1 = *ldbb;
    bb_offset = 1 + bb_dim1;
    bb -= bb_offset;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --w;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;
    --rwork;
    --iwork;
    --ifail;

    /* Function Body */
    wantz = lsame_(jobz, "V");
    upper = lsame_(uplo, "U");
    alleig = lsame_(range, "A");
    valeig = lsame_(range, "V");
    indeig = lsame_(range, "I");

    *info = 0;
    if (! (wantz || lsame_(jobz, "N"))) {
	*info = -1;
    } else if (! (alleig || valeig || indeig)) {
	*info = -2;
    } else if (! (upper || lsame_(uplo, "L"))) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*ka < 0) {
	*info = -5;
    } else if (*kb < 0 || *kb > *ka) {
	*info = -6;
    } else if (*ldab < *ka + 1) {
	*info = -8;
    } else if (*ldbb < *kb + 1) {
	*info = -10;
    } else if (*ldq < 1 || wantz && *ldq < *n) {
	*info = -12;
    } else {
	if (valeig) {
	    if (*n > 0 && *vu <= *vl) {
		*info = -14;
	    }
	} else if (indeig) {
	    if (*il < 1 || *il > f2cmax(1,*n)) {
		*info = -15;
	    } else if (*iu < f2cmin(*n,*il) || *iu > *n) {
		*info = -16;
	    }
	}
    }
    if (*info == 0) {
	if (*ldz < 1 || wantz && *ldz < *n) {
	    *info = -21;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZHBGVX", &i__1);
	return;
    }

/*     Quick return if possible */

    *m = 0;
    if (*n == 0) {
	return;
    }

/*     Form a split Cholesky factorization of B. */

    kpbstf_(uplo, n, kb, &bb[bb_offset], ldbb, info);
    if (*info != 0) {
	*info = *n + *info;
	return;
    }

/*     Transform problem to standard eigenvalue problem. */

    khbgst_(jobz, uplo, n, ka, kb, &ab[ab_offset], ldab, &bb[bb_offset], ldbb,
	     &q[q_offset], ldq, &work[1], &rwork[1], &iinfo);

/*     Solve the standard eigenvalue problem. */
/*     Reduce Hermitian band matrix to tridiagonal form. */

    indd = 1;
    inde = indd + *n;
    indrwk = inde + *n;
    indwrk = 1;
    if (wantz) {
	*(unsigned char *)vect = 'U';
    } else {
	*(unsigned char *)vect = 'N';
    }
    khbtrd_(vect, uplo, n, ka, &ab[ab_offset], ldab, &rwork[indd], &rwork[
	    inde], &q[q_offset], ldq, &work[indwrk], &iinfo);

/*     If all eigenvalues are desired and ABSTOL is less than or equal */
/*     to zero, then call DSTERF or ZSTEQR.  If this fails for some */
/*     eigenvalue, then try DSTEBZ. */

    test = FALSE_;
    if (indeig) {
	if (*il == 1 && *iu == *n) {
	    test = TRUE_;
	}
    }
    if ((alleig || test) && *abstol <= 0.) {
	hcopy_(n, &rwork[indd], &c__1, &w[1], &c__1);
	indee = indrwk + (*n << 1);
	i__1 = *n - 1;
	hcopy_(&i__1, &rwork[inde], &c__1, &rwork[indee], &c__1);
	if (! wantz) {
	    hsterf_(n, &w[1], &rwork[indee], info);
	} else {
	    klacpy_("A", n, n, &q[q_offset], ldq, &z__[z_offset], ldz);
	    ksteqr_(jobz, n, &w[1], &rwork[indee], &z__[z_offset], ldz, &
		    rwork[indrwk], info);
	    if (*info == 0) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    ifail[i__] = 0;
/* L10: */
		}
	    }
	}
	if (*info == 0) {
	    *m = *n;
	    goto L30;
	}
	*info = 0;
    }

/*     Otherwise, call DSTEBZ and, if eigenvectors are desired, */
/*     call ZSTEIN. */

    if (wantz) {
	*(unsigned char *)order = 'B';
    } else {
	*(unsigned char *)order = 'E';
    }
    indibl = 1;
    indisp = indibl + *n;
    indiwk = indisp + *n;
    hstebz_(range, order, n, vl, vu, il, iu, abstol, &rwork[indd], &rwork[
	    inde], m, &nsplit, &w[1], &iwork[indibl], &iwork[indisp], &rwork[
	    indrwk], &iwork[indiwk], info);

    if (wantz) {
	kstein_(n, &rwork[indd], &rwork[inde], m, &w[1], &iwork[indibl], &
		iwork[indisp], &z__[z_offset], ldz, &rwork[indrwk], &iwork[
		indiwk], &ifail[1], info);

/*        Apply unitary matrix used in reduction to tridiagonal */
/*        form to eigenvectors returned by ZSTEIN. */

	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    kcopy_(n, &z__[j * z_dim1 + 1], &c__1, &work[1], &c__1);
	    kgemv_("N", n, n, &c_b2, &q[q_offset], ldq, &work[1], &c__1, &
		    c_b1, &z__[j * z_dim1 + 1], &c__1);
/* L20: */
	}
    }

L30:

/*     If eigenvalues are not in order, then sort them, along with */
/*     eigenvectors. */

    if (wantz) {
	i__1 = *m - 1;
	for (j = 1; j <= i__1; ++j) {
	    i__ = 0;
	    tmp1 = w[j];
	    i__2 = *m;
	    for (jj = j + 1; jj <= i__2; ++jj) {
		if (w[jj] < tmp1) {
		    i__ = jj;
		    tmp1 = w[jj];
		}
/* L40: */
	    }

	    if (i__ != 0) {
		itmp1 = iwork[indibl + i__ - 1];
		w[i__] = w[j];
		iwork[indibl + i__ - 1] = iwork[indibl + j - 1];
		w[j] = tmp1;
		iwork[indibl + j - 1] = itmp1;
		kswap_(n, &z__[i__ * z_dim1 + 1], &c__1, &z__[j * z_dim1 + 1],
			 &c__1);
		if (*info != 0) {
		    itmp1 = ifail[i__];
		    ifail[i__] = ifail[j];
		    ifail[j] = itmp1;
		}
	    }
/* L50: */
	}
    }

    return;

/*     End of ZHBGVX */

} /* khbgvx_ */

