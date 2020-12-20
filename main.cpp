#include "Aprentissage.h"

int main()
{
    srand(time(NULL));
    std::string parameterAddress{""};
    std::cout<<"fichier pour les parametres si il existe, rien sinon"<<std::endl;
    std::cin>>parameterAddress;
    std::ifstream flux(parameterAddress, std::ios::in);
    std::istream *fluxx(0);
    if(flux)
    {
        fluxx=&flux;
    }
    else
    {
        fluxx=&std::cin;
    }

    Apprentissage apprentissage{*fluxx};
    apprentissage.learn();

    return 0;
}
