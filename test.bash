if ! { conda env list | grep 'py311'; } >/dev/null 2>&1; 
then
    echo "py311 doesn't exist!"
else
    echo 'py311 exists!'
fi
