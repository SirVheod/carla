#!/bin/sh

#This script builds debian package for CARLA
#Working with Ubuntu 18.04

#Check if Carla-<version> release is already downloaded
FILE=$(pwd)/carla-debian/carla-0.9.7/ImportAssets.sh #replace carla-0.9.7 with your desired carla-<version>
if [ -f "$FILE" ]; then
    ImportAssetscheck=1
else 
    ImportAssetscheck=0
fi

sudo apt-get install build-essential dh-make
mkdir carla-debian/
mkdir carla-debian/carla-0.9.7
cd carla-debian/carla-0.9.7


#If carla release is not already downloaded then it will download
if [ ${ImportAssetscheck} == 0 ]
then
	#replace below urls with your desired carla release urls
	curl http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.7.tar.gz | tar xz
	wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/AdditionalMaps_0.9.7.tar.gz
	
	mv AdditionalMaps_0.9.7.tar.gz Import/
	
fi

./ImportAssets.sh  #importing new maps 

#removing unnecessary files
rm CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping.debug
rm CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping.sym

rm CarlaUE4.sh
cat >> CarlaUE4.sh <<EOF
#!/bin/sh

chmod +x "/opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
"/opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" CarlaUE4 $@
EOF

dh_make --indep --createorig -y  #to create necessary file structure for debian packaging

cd debian/

#removing unnecessary files
rm *.ex
rm *.EX

rm control

#Adding package dependencies(Package will install them itself) and description
cat >> control <<EOF
Source: carla
Section: simulator
Priority: optional
Maintainer: Carla Simulator Team <carla.simulator@gmail.com>
Build-Depends: debhelper (>= 10)
Standards-Version: 0.9.7
Homepage: http://carla.org/

Package: carla
Architecture: any
Depends: python,
         python-numpy,
         python-pygame,
         libpng16-16,
         libjpeg8,
         libtiff5
Description: Open-source simulator for autonomous driving research
 CARLA has been developed from the ground up to support development, training, and validation
 of autonomous driving systems. In addition to open-source code and protocols, CARLA provides 
 open digital assets (urban layouts, buildings, vehicles) that were created for this purpose 
 and can be used freely. The simulation platform supports flexible specification of sensor suites, 
 environmental conditions, full control of all static and dynamic actors, maps generation and much more.
EOF

rm install

#Making debian package to install Carla in /opt folder
cat>> install <<EOF
CarlaUE4.sh /opt/carla/bin
ImportAssets.sh /opt/carla
CarlaUE4 /opt/carla
Engine /opt/carla
Import /opt/carla
PythonAPI /opt/carla

EOF

rm postinst

#Adding Carla library path (carla.pth) to site-packages, during post installation.
#Change Carla version according to need
cat>> postinst << EOF
#!/bin/sh

SITEDIR=\$(python3 -c 'import site; site._script()' --user-site)
mkdir -p "\$SITEDIR"
echo "/opt/carla/PythonAPI/carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg\n/opt/carla/PythonAPI/carla/" > "\$SITEDIR/carla.pth"

SITEDIR=\$(python2 -c 'import site; site._script()' --user-site)
mkdir -p "\$SITEDIR"
echo "/opt/carla/PythonAPI/carla/dist/carla-0.9.7-py2.7-linux-x86_64.egg\n/opt/carla/PythonAPI/carla/" > "\$SITEDIR/carla.pth"

chmod +x /opt/carla/bin/CarlaUE4.sh

set -e

case "\$1" in
    configure)
    ;;

    abort-upgrade|abort-remove|abort-deconfigure)
    ;;

    *)
        echo "postinst called with unknown argument \\\`\$1'" >&2
        exit 1
    ;;
esac

exit 0

EOF

rm prerm

#Removing Carla library from site-packages
cat>> prerm << EOF
#!/bin/sh

SITEDIR=\$(python3 -c 'import site; site._script()' --user-site)
rm "\$SITEDIR/carla.pth"

SITEDIR=\$(python2 -c 'import site; site._script()' --user-site)
rm "\$SITEDIR/carla.pth"


set -e

case "\$1" in
    remove|upgrade|deconfigure)
    ;;

    failed-upgrade)
    ;;

    *)
        echo "prerm called with unknown argument \\\`\$1'" >&2
        exit 1
    ;;
esac

exit 0

EOF

cd ..
debuild --no-lintian -uc -us -b #building debian package
cd ..
# install debian package using "sudo dpkg -i carla_0.9.7-1_amd64.deb"