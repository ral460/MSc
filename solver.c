#include <math.h>
#include <string.h>
#include "Rinternals.h"
#include "R_ext/Rdynload.h"
#include <R.h>
#include <R_ext/Applic.h>

// Gaussian loss
double gLoss(double *r, int n) {
  double l = 0;
  for (int i=0;i<n;i++) l = l + pow(r[i],2);
  return(l);
}

// Memory handling, output formatting (Gaussian)
SEXP cleanupRawG(double *a, double *v, double *u, int *active, SEXP beta, SEXP loss, SEXP iter, SEXP resid) {
  Free(a);
  Free(v);
  Free(u);
  Free(active);
  SEXP res;
  PROTECT(res = allocVector(VECSXP, 4));
  SET_VECTOR_ELT(res, 0, beta);
  SET_VECTOR_ELT(res, 1, loss);
  SET_VECTOR_ELT(res, 2, iter);
  SET_VECTOR_ELT(res, 3, resid);
  UNPROTECT(1);
  return(res);
}

// Group-wise Coordinate descent for gaussian models
SEXP rawfit_glasso(SEXP X_, SEXP y_, SEXP INIT_, SEXP n_est_, SEXP R_, SEXP maxeigen_, SEXP lambda, SEXP eps_, SEXP max_iter_, SEXP multiplier, SEXP alpha_) {

  // Declarations: Outcome
  int const n_est = INTEGER(n_est_)[0];
  int const n_y = length(y_);
  int const n = n_y/n_est;
  int const p_design = length(X_)/n;
  int const p = p_design/n_est;
  SEXP res, BETA, loss, iter, resid;
  PROTECT(BETA = allocVector(REALSXP, p_design));
  double *B = REAL(BETA);
  for (int j=0; j<p_design; j++) B[j] = 0;
  PROTECT(loss = allocVector(REALSXP, 1));
  PROTECT(iter = allocVector(INTSXP, 1));
  PROTECT(resid = allocVector(REALSXP, n_y));
  INTEGER(iter)[0] = 0;
  
  // Declarations
  double *X = REAL(X_);
  double *y = REAL(y_);
  double *A = Calloc(p_design, double); // BETA from previous iteration
  for (int j=0; j<p_design; j++) A[j]=REAL(INIT_)[j];
  double lam = REAL(lambda)[0];
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier);
  int *active = Calloc(p, int);
  for (int j=0; j<p; j++){
    double normB = 0;
    for (int l=0; l<n_est; l++){
      normB += pow(A[l*p+j],2);
      active[j] = 1*(normB != 0);
    }
  }

  // Setup R, v, Z
  double *R = REAL(resid);
  for (int i=0; i<n_y; i++) R[i] = y[i];    
  for (int j=0; j<p; j++) {
    for (int i=0; i<n; i++) {
      for (int l=0; l<n_est; l++) {
        R[l*n+i] -= X[(l*p+j)*n+i]*A[l*p+j];
      }
    }
  }
  
  double *v = Calloc(p, double);
  for (int j=0; j<p; j++) v[j] = REAL(maxeigen_)[j];
  double *U = Calloc(p_design, double);
  double sdy = sqrt(gLoss(y, n_y)/n_y); //

  // Fit
  while (INTEGER(iter)[0] < max_iter) {
    R_CheckUserInterrupt();
    while (INTEGER(iter)[0] < max_iter) {
      INTEGER(iter)[0]++;
      // Solve over each group j
      double maxChange = 0;
      for (int j=0; j<p; j++) {
        double normUB = 0;
        if (active[j]) {
          for(int l=0; l<n_est; l++){
            U[l*p+j]=0;
            for (int i=0;i<n;i++) U[l*p+j] += X[(l*p+j)*n+i]*R[l*n+i];
            U[l*p+j] = U[l*p+j]/n_y;
            normUB += pow(U[l*p+j] + v[j]*A[l*p+j], 2); 
          }
          
          for(int l=0; l<n_est; l++){
            B[l*p+j] = fmax(1-lam * m[j] / sqrt(normUB), 0) *  (U[l*p+j]/v[j] + A[l*p+j]);  
            double shift = B[l*p+j] - A[l*p+j];
            for (int i=0; i<n; i++){
              R[l*n+i] -= shift*X[(l*p+j)*n+i]; 
              if (fabs(shift)*sqrt(v[j]) > maxChange) maxChange = fabs(shift) * sqrt(v[j]);
            }
          } 
        }
      } //End for
        
      // Check for convergence
      for (int j=0; j<p_design; j++) A[j] = B[j]; 
      if (maxChange < eps*sdy) break;
    } //End While
    // Scan for violations
    int violations = 0;
    for (int j=0; j<p; j++) {
        double normUB = 0;
        if (!active[j]) {
          for(int l=0; l<n_est; l++){
            U[l*p+j]=0;
            for (int i=0;i<n;i++) U[l*p+j] += X[(l*p+j)*n+i]*R[l*n+i];
            U[l*p+j] = U[l*p+j]/n_y;
            normUB += pow(U[l*p+j] + v[j]*A[l*p+j], 2); 
          }
          
          for(int l=0; l<n_est; l++){
            B[l*p+j] = fmax(1-lam * m[j] / sqrt(normUB), 0) *  (U[l*p+j]/v[j] + A[l*p+j]);  
            if (B[l*p+j]) {
              active[j] = 1;
              for (int i=0; i<n; i++) R[l*n+i] -= B[l*p+j] * X[(l*p+j)*n+i]; 
              A[l*p+j] = B[l*p+j];
              violations++;
            }
          } 
        }
      }

    if (violations==0) break;
  }
  REAL(loss)[0] = gLoss(R, n); // CHECK HERE
  res = cleanupRawG(A, v, U, active, BETA, loss, iter, resid);
  UNPROTECT(4);
  return(res);
}
