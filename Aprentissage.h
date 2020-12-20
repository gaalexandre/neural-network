#ifndef APRENTISSAGE_H
#define APRENTISSAGE_H

#include "NeuralNetwork.h"
#include <thread>
#include "Database.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


class Apprentissage
{

public:
    Apprentissage(std::istream &flux);
    ~Apprentissage();
    void learn();
    double test();
private:
    void createDataBase(std::string type, std::string dataAddress);
    void setParameters();

    void choisirCostFunction();
    void choisirActFunction();
    void choisir(std::string texte, int *&entier);
    void choisir(std::string texte, double *&flottant);

    void stop();
    bool m_stop{false};

    std::istream &m_flux;
    int m_nbNetwork{1};
    int m_nbTread{1};

    Database *m_data{0};
    int const* m_nbTestExemple{0};

    int *m_nbLayer = new int[3];
    int **m_nbNeuron{0};
    double **m_learningRate{0};
    int *m_miniBatchSize = new int[3];
    int *m_nbEpoch = new int[3];
    bool m_save;
    std::string m_saveAddress;

    CostFunction *m_costFunction{0};
    ActFunction **m_actFunction{0};



    class TrainSet
    {
    public:
        TrainSet();
        ~TrainSet();
        void init(Database const* data, int *nbLayer, int **nbNeuron, double **learningRate, int *miniBatchSize, int *nbEpoch
        , CostFunction *costFunction, ActFunction **actFunction,bool *stop,bool save, int id,std::string saveAddress);
        void trainNetwork();
        double validation();
        void setSave(bool save);

    private:

        int evaluation(int *entier);
        double evaluation(double *flottant);

        inline void feedForward();
        inline void calculOutputError();
        inline void backpropagation();
        inline void gradientDescend();

        void resizeMiniBatch(int miniBatchSize);

        void save();

        bool m_save;
        int m_id;
        std::string m_saveAddress;

        int m_nbLayer;
        int *m_nbNeuron{0};

        CostFunction const*m_costFunction{0};
        ActFunction const*const*m_actFunction{0};

        double *m_learningRate{0};
        int m_miniBatchSize;

        Eigen::MatrixXd *m_sortieAttendue{0};
        Eigen::MatrixXd *m_error{0};
        NeuralNetwork *m_neuralNetwork{0};

        int m_nbEpoch;

        double m_validationScore{0};

        Database const* m_data{0};
        int const* m_nbTrainingExemple{0};
        int const* m_nbValidationExemple{0};

        bool const* m_stop{0};
    };

    TrainSet *m_bestTrainSet{0};
    int m_bestValidation{0};

};

#endif // APRENTISSAGE_H
