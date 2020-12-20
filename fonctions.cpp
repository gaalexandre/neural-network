#include "fonctions.h"


std::string intToString(int n)
{
    std::ostringstream str;

    str << n;

    return str.str();
}

Eigen::MatrixXd const UpTanh::operator() (Eigen::MatrixXd const&mat) const
{
    return 1.7159*((2./3)*mat).array().tanh();
}

Eigen::MatrixXd const UpTanh::prime(Eigen::MatrixXd const&mat) const
{
    return 1.7159*2/3.*((2/3.)*mat).array().cosh().pow(2).inverse();
}

std::string UpTanh::nom() const
{
    return "UpTanh";
}

Eigen::MatrixXd const Tanh::operator() (Eigen::MatrixXd const&mat) const
{
    return mat.array().tanh();
}

Eigen::MatrixXd const Tanh::prime(Eigen::MatrixXd const&mat) const
{
    return mat.array().cosh().pow(2).inverse();
}

std::string Tanh::nom() const
{
    return "Tanh";
}

Eigen::MatrixXd const Sigmoid::operator() (Eigen::MatrixXd const&mat) const
{
    return ((-mat.array()).exp()+1).inverse();
}

Eigen::MatrixXd const Sigmoid::prime(Eigen::MatrixXd const&mat) const
{
    return (mat.array().exp()+2+(-mat).array().exp()).inverse();
}

std::string Sigmoid::nom() const
{
    return "Sigmoid";
}

Eigen::MatrixXd const ReLU::operator() (Eigen::MatrixXd const&mat) const
{
    return mat.cwiseMax(0.01*mat);
}

Eigen::MatrixXd const ReLU::prime(Eigen::MatrixXd const&mat) const
{
    return mat.array().unaryExpr([](double x) { if(x>=0)return 1.;else return 0.01;});
}

std::string ReLU::nom() const
{
    return "ReLU";
}

Eigen::MatrixXd const SoftMax::operator() (Eigen::MatrixXd const&mat) const
{
    return mat.array().exp().rowwise()*mat.array().exp().colwise().sum().inverse();
}

Eigen::MatrixXd const SoftMax::prime(Eigen::MatrixXd const&mat) const
{
    auto a=(this->operator()(mat)).array();
    return a-a.pow(2);
}

std::string SoftMax::nom() const
{
    return "SoftMax";
}

double CrossEntropy::operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const
{
    return -1/(output.cols())*((desiredOutput.array()*Eigen::log(output.array())+(1-desiredOutput.array())*Eigen::log(1-output.array())).sum());
}

Eigen::MatrixXd const CrossEntropy::gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const
{
    return -1/(output.cols())*((desiredOutput.array()/output.array() - (1-desiredOutput.array())/(1-output.array()))).rowwise().sum();
}

std::string CrossEntropy::nom() const
{
    return "CrossEntropy";
}

double Quadratic::operator() (Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const
{
    double norme=(output-desiredOutput).norm();
    return (norme*norme)/output.cols()/2;
}

Eigen::MatrixXd const Quadratic::gradient(Eigen::MatrixXd const&output,Eigen::MatrixXd const&desiredOutput) const
{
    return (output-desiredOutput)/output.cols();
}

std::string Quadratic::nom() const
{
    return "Quadratic";
}
