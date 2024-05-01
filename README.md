# SFML_Template

### How to compile your application

Use the installation script to launch the building and compiling processes

```shell
sh install.sh
```

Then the executable of your application will appear in the same directory

### How to use this template

As a developper, the only repository you'll need is the **<u>src/Application</u>** directory where you can find the source file **App.cpp** and his header **App.hpp**

> This file contains two functions which are the core of your application,
> 
> - ***App::update( sf::Time deltaTime )*** - This function is called every frame by a while loop located in the **<u>src/Engine</u>** directory, in the file **Engine.cpp**. It allows you to change the state of your application
> 
> - ***App::draw( sf::RenderTarget& target, sf::RenderStates states )*** - This function is also called every frame and must be used to draw SFML objects to the screen

You can edit these files and add any numbers of sources files you want in <u>**src/Application**</u>, it is where you will developp your app. 

### How to start a new project with this template

Create a bare clone of the template

```shell
git clone --bare https://github.com/MHAVGOUDOUKIAN/SFML_Template.git
```

Push the mirror to the new repository

```shell
cd OLD-REPOSITORY.git
git push --mirror https://github.com/MHAVGOUDOUKIAN/NEW-SFML-PROJECT.git
```

Delete the temporary local repository you created

```shell
cd ..
rm -rf OLD-REPOSITORY.git
```
