#include <Tests/Matrix_test.hpp>

void launch_test() {

    // Testing - Matrix::Matrix(std::size_t row, std::size_t col)
    int nb_row{5}, nb_col{5};
    Matrix A {Matrix(nb_row, nb_col)};
    
    assert(A.col()==nb_col && A.row()==nb_row);

    for(int i=0; i<A.row(); i++) {  
        for(int j=0; j<A.col(); j++) {
            assert(A.getCoeff(i,j) >= 0.0f && A.getCoeff(i,j) <= 1.0f);
        }
    }

    nb_row = 30, nb_col = 12;
    A = Matrix(nb_row, nb_col);
    
    assert(A.col()==nb_col && A.row()==nb_row);

    for(int i=0; i<A.row(); i++) {  
        for(int j=0; j<A.col(); j++) {
            assert(A.getCoeff(i,j) >= 0.0f && A.getCoeff(i,j) <= 1.0f);
        }
    }

    // Testing - Matrix::Matrix(std::size_t row, std::size_t col, float value)
    nb_row == 5, nb_col == 5;
    float value = 6.f;
    A = Matrix(nb_row, nb_col, value);
    
    assert(A.col()==nb_col && A.row()==nb_row);

    for(int i=0; i<A.row(); i++) {  
        for(int j=0; j<A.col(); j++) {
            assert(A.getCoeff(i,j) == value);
        }
    }

    nb_row = 30, nb_col = 12, value=-5.f;
    A = Matrix(nb_row, nb_col, value);
    
    assert(A.col()==nb_col && A.row()==nb_row);

    for(int i=0; i<A.row(); i++) {  
        for(int j=0; j<A.col(); j++) {
            assert(A.getCoeff(i,j) == value);
        }
    }

    // Testing - Matrix Matrix::operator+(const Matrix& B) const
    A = Matrix(5,5,value);
    Matrix B{Matrix(5,5)}, res{A+B};

    for(int i=0; i<res.row(); i++) {
        for(int j=0; j<res.col(); j++) {
            assert(res.getCoeff(i,j) == A.getCoeff(i,j) + B.getCoeff(i,j));
        }
    }

    // Testing - Matrix Matrix::operator*(const Matrix& B) const
    A = Matrix(5,2), B=Matrix(2,3), res=A*B;
    std::cout << "TEST operator*Matrix" << std::endl;
    A.disp();
    B.disp();
    res.disp();

    // Testing - void Matrix::operator-=(const Matrix& B)
    A = Matrix(5,7), B=Matrix(5,7), res=A;
    res-=B;

    for(int i=0; i<res.row(); i++) {
        for(int j=0; j<res.col(); j++) {
            assert(res.getCoeff(i,j) == A.getCoeff(i,j)-B.getCoeff(i,j));
        }
    }

    // Testing - Matrix Matrix::transposee() const
    A = Matrix(5,7), B=A.transposee();
    for(int i=0; i<res.row(); i++) {
        for(int j=0; j<res.col(); j++) {
            assert(A.getCoeff(i,j)==B.getCoeff(j,i));
        }
    }

    // Testing - Matrix Hadamard(const Matrix& A, const Matrix& B)
    A = Matrix(5,7), B = Matrix(5,7), res = Hadamard(A,B);
    for(int i=0; i<res.row(); i++) {
        for(int j=0; j<res.col(); j++) {
            assert(res.getCoeff(i,j) == A.getCoeff(i,j)*B.getCoeff(i,j));
        }
    }

    // Testing - void Matrix::operator+(const float value) 
    A = Matrix(5,7);
    value = 5.f;
    std::cout << "TEST operator+value:" << value << std::endl;
    A.disp();
    A+value;
    A.disp();
    // Testing - void Matrix::operator+(const float value) 
    value = 2.f;
    std::cout << "TEST operator*value:" << value << std::endl;
    A*value;
    A.disp();

    // Testing - Matrix SumOnCol(const Matrix& A)
    std::cout << "TEST SumOnCol:" << std::endl;
    A = Matrix(5,7);
    A.disp();
    A = SumOnCol(A);
    A.disp();

}