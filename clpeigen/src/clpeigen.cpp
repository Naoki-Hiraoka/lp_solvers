#include <clpeigen/clpeigen.h>

namespace clpeigen{
  solver::solver():
    initialized(false),
    initial_solve(true)
  {
  }

  bool solver::initialize(const Eigen::VectorXd& o,
                          const Eigen::SparseMatrix<double,Eigen::RowMajor>& A,
                          const Eigen::VectorXd& lbA,
                          const Eigen::VectorXd& ubA,
                          const Eigen::VectorXd& lb,
                          const Eigen::VectorXd& ub,
                          int debuglevel){
    int numberRows = A.rows();
    int numberColumns = A.cols();
    int numberElements = A.nonZeros();//0の数ではなく要素数を見ている

    // matrix data - row ordered
    std::vector<int> len(A.rows());
    for(size_t i=0;i<A.rows();i++){
      len[i] = A.row(i).nonZeros();
    }
    CoinPackedMatrix matrix(false,//true: ColMajor, false: RowMajor
                            numberColumns,//minor
                            numberRows,//major
                            numberElements,
                            A.valuePtr(),
                            A.innerIndexPtr(),
                            A.outerIndexPtr(),
                            len.data());

    // load problem
    this->model_.loadProblem(matrix,
                            lb.data(),
                            ub.data(),
                            o.data(),
                            lbA.data(),
                            ubA.data());
    this->model_.setOptimizationDirection(-1);//maximize
    this->model_.setLogLevel(debuglevel);

    this->initialized = true;
    this->initial_solve = true;
    return true;
  }

  bool solver::solve(){
    // Solve
    if(this->initial_solve){
      int status = this->model_.initialSolve();
      this->initial_solve = false;
      return status == 0;
    }else{
      int status = this->model_.initialSolve();
      //int status = this->model_.primal(1);// 安定しない
      //int status = this->model_.barrier(); // 安定しない
      return status == 0;
    }
  }

  bool solver::getSolution(Eigen::VectorXd& solution){
    solution.resize(this->model_.getNumCols());

    // Solution
    const double * s = this->model_.primalColumnSolution();
    for (size_t i=0; i < this->model_.getNumCols(); i++) solution[i] = s[i];

    return true;
  }

  bool solver::updateObjective(const Eigen::VectorXd& o){
    if(o.rows() != this->model_.getNumCols()){
      std::cerr << "[clpeigen::solver::updateObjective] dimention mismatch" << std::endl;
      return false;
    }

    double * objective = this->model_.objective();
    for(size_t i=0;i<this->model_.getNumCols();i++) objective[i] = o[i];

    return true;
  }
}
