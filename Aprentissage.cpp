#include "Aprentissage.h"

Apprentissage::Apprentissage(std::istream &flux):m_flux(flux)
{
    std::string dataAddress;
    std::cout<<"dataAddress"<<std::endl;
    m_flux>>dataAddress;

    std::cout<<"nbNetwork <-100 pour infinit"<<std::endl;
    m_flux>>m_nbNetwork;

    std::cout<<"nbThread"<<std::endl;
    m_flux>>m_nbTread;

    std::ifstream data(dataAddress, std::ios::in);

    if(data)
    {
        std::string type;
        data>>type;

        createDataBase(type, dataAddress);
        data.close();
    }
    else
        std::cerr << "data pas ouverte" << std::endl;

    m_nbTestExemple=m_data->getNbTestExemple();
    m_saveAddress=dataAddress+"aa\\";
    setParameters();
}

Apprentissage::~Apprentissage()
{
    delete m_data;
    for (int i{0};i<m_nbLayer[2];i++)
    {
        delete[] m_nbNeuron[i];
        delete[] m_learningRate[i];
        delete m_actFunction[i];
    }
    delete[] m_nbNeuron;
    delete[] m_learningRate;

    delete[] m_miniBatchSize;
    delete[] m_nbEpoch;

    delete m_costFunction;
    delete[] m_nbLayer;
    delete[] m_actFunction;

    delete m_bestTrainSet;
}

void Apprentissage::learn()
{
    std::thread threadStop(&Apprentissage::stop,this);
    for(int j{0};j!=m_nbNetwork/m_nbTread&&!m_stop;j++)
    {
        std::thread threads[m_nbTread];

        TrainSet* trainSet[m_nbTread];

        for(int i{0};i<m_nbTread;i++)
        {
            trainSet[i]=new TrainSet;
            trainSet[i]->init(m_data, m_nbLayer, m_nbNeuron, m_learningRate, m_miniBatchSize, m_nbEpoch, m_costFunction, m_actFunction, &m_stop,m_save,j*m_nbTread+i,m_saveAddress);
            threads[i]=std::thread(&Apprentissage::TrainSet::trainNetwork,trainSet[i]);

        }
        for(int i{0};i<m_nbTread;i++)
        {
            threads[i].join();
            double valid(trainSet[i]->validation());
            if(valid<m_bestValidation||m_bestTrainSet==0)
            {
                m_bestValidation=valid;
                if(m_bestTrainSet!=0)
                    m_bestTrainSet->setSave(m_save);
                delete m_bestTrainSet;
                m_bestTrainSet=trainSet[i];
                m_bestTrainSet->setSave(true);
                trainSet[i]=0;
            }
            delete trainSet[i];
        }
    }
    m_stop=true;
    threadStop.join();

}

void Apprentissage::createDataBase(std::string type, std::string dataAddress)
{

    if(type=="bool")
        m_data=new DatabaseT<bool>{dataAddress};
    else if(type=="char")
        m_data=new DatabaseT<char>{dataAddress};
    else if(type=="unsignedChar")
        m_data=new DatabaseT<unsigned char>{dataAddress};
    else if(type=="shortInt")
        m_data=new DatabaseT<short int>{dataAddress};
    else if(type=="unsignedShortInt")
        m_data=new DatabaseT<unsigned short int>{dataAddress};
    else if(type=="int")
        m_data=new DatabaseT<int>{dataAddress};
    else if(type=="unsignedInt")
        m_data=new DatabaseT<unsigned int>{dataAddress};
    else if(type=="longInt")
        m_data=new DatabaseT<long int>{dataAddress};
    else if(type=="unsignedLongInt")
        m_data=new DatabaseT<unsigned long int>{dataAddress};
    else if(type=="longLongInt")
        m_data=new DatabaseT<long long int>{dataAddress};
    else if(type=="unsignedLongLongInt")
        m_data=new DatabaseT<unsigned long long int>{dataAddress};
    else if(type=="float")
        m_data=new DatabaseT<float>{dataAddress};
    else if(type=="double")
        m_data=new DatabaseT<double>{dataAddress};
    else if(type=="longDouble")
        m_data=new DatabaseT<long double>{dataAddress};
}

void Apprentissage::setParameters()
{
    choisir("m_nbLayer", m_nbLayer);
    m_nbNeuron = new int*[m_nbLayer[2]];
    m_nbNeuron[0]=new int[3];
    m_nbNeuron[m_nbLayer[2]-1]=new int[3];
    for (int i{0};i<3;i++)
    {
        m_nbNeuron[m_nbLayer[2]-1][i]=m_data->getOutputSize();
        m_nbNeuron[0][i]=m_data->getInputSize();
    }
    for (int i{1};i<m_nbLayer[2]-1;i++)
    {
        m_nbNeuron[i] = new int[3];
        std::cout<<"Initialisation de m_nbNeuron dans la couche eventuelle "<< i<<std::endl;
        choisir("m_nbNeuron", m_nbNeuron[i]);
    }
    m_learningRate = new double*[m_nbLayer[2]];
    m_learningRate[0]=0;
    for (int i{1};i<m_nbLayer[2];i++)
    {
        m_learningRate[i] = new double[3];
        std::cout<<"Initialisation de m_learningRate dans la couche eventuelle "<< i<<std::endl;
        choisir("m_learningRate", m_learningRate[i]);
    }
    choisir("m_miniBatchSize", m_miniBatchSize);
    choisir("m_nbEpoch", m_nbEpoch);
    int *save=new int[3];
    choisir("m_save(tout sauvegarder ou non) 0/1 ", save);
    m_save=(bool)save[0];
    delete[] save;
    choisirCostFunction();
    choisirActFunction();

}

void Apprentissage::stop()
{
    std::string st;
    while(!m_stop)
    {
        std::cout<<"taper stop pour stopper"<<std::endl;
        std::cin>>st;
        if(st=="stop")
            m_stop=true;
    }
}



double Apprentissage::TrainSet::evaluation(double *flottant)
{
    if(flottant[0]<0)
    {
        return rand()/(double) RAND_MAX  * (flottant[2]-flottant[1])+flottant[1];
    }
    else
        return flottant[0];
}

 int Apprentissage::TrainSet::evaluation(int *entier)
{
    if(entier[0]<0)
    {
        return rand()%(entier[2]-entier[1])+entier[1];
    }
    else
        return entier[0];
}

void Apprentissage::choisir(std::string texte, int *&entier)
{
    std::cout<<"Entrez "<< texte <<" - negatif pour aleatoire"<<std::endl;
    m_flux>>entier[0];
    if (entier[0]<0)
    {
        std::cout<<"Entrez l'intervalle dans le cas aleatoire, 2 entiers arbitraires sinon"<<std::endl;
        m_flux>>entier[1]>>entier[2];
    }
    else
    {
        entier[1] = entier[0];
        entier[2] = entier[0];
    }
}

void Apprentissage::choisir(std::string texte, double *&flottant)
{
    std::cout<<"Entrez "<< texte <<" - negatif pour aleatoire"<<std::endl;
    m_flux>>flottant[0];
    if (flottant[0]<0)
    {
        std::cout<<"Entrez l'intervalle dans le cas aleatoire, 2 flottants arbitraires sinon"<<std::endl;
        m_flux>>flottant[1]>>flottant[2];
    }
    else
    {
        flottant[1] = flottant[0];
        flottant[2] = flottant[0];
    }
}

void Apprentissage::choisirCostFunction()
{
    std::cout<<"Entrez la fonction de cout :"<< std::endl;
    std::cout<<"1 pour CrossEntropy, 2 pour Quadratic"<<std::endl;
    int idCostFunction;
    m_flux>>idCostFunction;
    if(idCostFunction==1)
        m_costFunction = new CrossEntropy;
    if(idCostFunction==2)
        m_costFunction = new Quadratic;

}

void Apprentissage::choisirActFunction()
{
    m_actFunction = new ActFunction*[m_nbLayer[2]];
    m_actFunction[0]=0;
    std::cout<<"Entrez les "<<m_nbLayer[2]-1<<"fonctions d'activation eventuelles:"<< std::endl;
    std::cout<<"1 pour Sigmoid, 2 pour SoftMax, 3 pour Tanh, 4 pour UpTanh, 5 pour ReLU"<<std::endl;
    for (int i{1};i<m_nbLayer[2];i++)
    {
        int idActFunction;
        m_flux>>idActFunction;
        if(idActFunction==1)
            m_actFunction[i] = new Sigmoid;
        if(idActFunction==2)
            m_actFunction[i] = new SoftMax;
        if(idActFunction==3)
            m_actFunction[i] = new Tanh;
        if(idActFunction==4)
            m_actFunction[i] = new UpTanh;
        if(idActFunction==5)
            m_actFunction[i] = new ReLU;
    }
}

Apprentissage::TrainSet::TrainSet(){}

Apprentissage::TrainSet::~TrainSet()
{
    if(m_save)
        save();
    delete[] m_nbNeuron;
    delete[] m_learningRate;
    delete m_sortieAttendue;
    delete[] m_error;
    delete m_neuralNetwork;
}

void Apprentissage::TrainSet::init(Database const* data, int *nbLayer, int **nbNeuron, double **learningRate, int *miniBatchSize, int *nbEpoch, CostFunction *costFunction, ActFunction **actFunction,bool *stop,bool save, int id, std::string saveAddress)
{
    m_stop=stop;
    m_save=save;
    m_id=id;
    m_saveAddress=saveAddress+"trainSet"+intToString(id);
    m_data=data;
    m_nbTrainingExemple=m_data->getNbTrainingExemple();
    m_nbValidationExemple=m_data->getNbValidationExemple();
    m_nbLayer = evaluation(nbLayer);
    m_nbNeuron = new int[m_nbLayer];
    for (int i{0}; i<m_nbLayer; i++)
        m_nbNeuron[i] = evaluation(nbNeuron[i]);
    m_learningRate = new double[m_nbLayer];
    for (int i{1}; i<m_nbLayer; i++)
        m_learningRate[i] = evaluation(learningRate[i]);
    m_miniBatchSize = evaluation(miniBatchSize);
    m_nbEpoch = evaluation(nbEpoch);
    m_costFunction = costFunction;
    m_actFunction = &actFunction[nbLayer[2]-m_nbLayer];
    m_sortieAttendue=new Eigen::MatrixXd(m_nbNeuron[m_nbLayer-1],m_miniBatchSize);
    m_error=new Eigen::MatrixXd[m_nbLayer];
    m_neuralNetwork=new NeuralNetwork(m_nbLayer,m_nbNeuron,m_actFunction,m_miniBatchSize,m_save,saveAddress+"neuralNetwork"+intToString(id));
    for(int i{0};i<m_nbLayer;i++)
        m_error[i]=m_neuralNetwork->m_layer[i];

}

double Apprentissage::TrainSet::validation()
{

    int miniBatchSize=m_miniBatchSize;
    resizeMiniBatch(*m_nbValidationExemple);
    m_data->loadValidationInput(m_neuralNetwork->m_layer[0],*m_sortieAttendue);
    m_neuralNetwork->calcul();
    double valid{(*m_costFunction)(m_neuralNetwork->m_layer[m_nbLayer-1],*m_sortieAttendue)};
    m_validationScore=valid;
    resizeMiniBatch(miniBatchSize);
    return valid;
}

void  Apprentissage::TrainSet::setSave(bool save)
{
    m_save=save;
    m_neuralNetwork->m_save=save;
}

void Apprentissage::TrainSet::trainNetwork()
{
    for(int j{0};j<m_nbEpoch&&!*m_stop;j++)
    {
        for(int k{0};k<*m_nbTrainingExemple;k+=m_miniBatchSize)
        {
            m_data->loadTrainingInput(m_neuralNetwork->m_layer[0],*m_sortieAttendue,k,m_miniBatchSize);
            feedForward();
            calculOutputError();
            backpropagation();
            gradientDescend();
        }

         double val(validation());
         std::cout<<val<<std::endl;
    }
}


void Apprentissage::TrainSet::feedForward()
{
    for(int j{1};j<m_nbLayer;j++)
    {
        m_error[j]=(m_neuralNetwork->m_weight[j]*m_neuralNetwork->m_layer[j-1]).colwise()+m_neuralNetwork->m_bias[j];
        m_neuralNetwork->m_layer[j]=(*m_actFunction[j])(m_error[j]);
    }
}

void Apprentissage::TrainSet::calculOutputError()
{
    m_error[m_nbLayer-1]=m_actFunction[m_nbLayer-1]->prime(m_error[m_nbLayer-1]).cwiseProduct(m_costFunction->gradient(m_neuralNetwork->m_layer[m_nbLayer-1],*m_sortieAttendue));
}

void Apprentissage::TrainSet::backpropagation()
{
    for(int j{m_nbLayer-2};j>0;j--)
    {
         m_error[j]=m_actFunction[j]->prime(m_error[j]).cwiseProduct(m_neuralNetwork->m_weight[j+1].transpose()*m_error[j+1]);
    }
}

void Apprentissage::TrainSet::gradientDescend()
{
    for(int j{m_nbLayer-1};j>0;j--)
    {
        m_neuralNetwork->m_bias[j]-=(m_learningRate[j]/m_miniBatchSize)*m_error[j].rowwise().sum();
        m_neuralNetwork->m_weight[j]-=(m_learningRate[j]/m_miniBatchSize)*(m_error[j]*((m_neuralNetwork->m_layer[j-1]).transpose()));
    }
}

void Apprentissage::TrainSet::resizeMiniBatch(int miniBatchSize)
{
    m_miniBatchSize=miniBatchSize;
    for(int i{0};i<m_nbLayer;i++)
    {
        m_error[i].resize(m_error[i].rows(),m_miniBatchSize);
        m_neuralNetwork->m_layer[i].resize(m_neuralNetwork->m_layer[i].rows(),m_miniBatchSize);
    }
    m_sortieAttendue->resize(m_sortieAttendue->rows(),m_miniBatchSize);
}

void Apprentissage::TrainSet::save()
{
    std::ofstream file(m_saveAddress+".txt", std::ios::out | std::ios::trunc);
    if(file)
    {
        file << m_data->nom() << std::endl;
        file << m_nbLayer << std::endl;
        file << m_costFunction->nom() << std::endl;
        for(int i{1};i<m_nbLayer;i++)
            file<<m_actFunction[i]->nom()<<" ";
        file<<std::endl;
        for(int i{0};i<m_nbLayer;i++)
            file<<m_nbNeuron[i]<<" ";
        file<<std::endl;
        for(int i{0};i<m_nbLayer;i++)
            file << m_learningRate[i] << " ";
        file<<std::endl;
        file << m_miniBatchSize << std::endl;
        file << m_nbEpoch << std::endl;
        file << m_validationScore << std::endl;

        file.close();
    }
    else
        std::cerr << "fichier non ouvert" << std::endl;
}
