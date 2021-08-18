rm ./dist/*
python3 setup.py check && python3 setup.py sdist && twine upload dist/*
python3 -m pip install --upgrade nodeclass && \
python3 -m pip install --upgrade nodeclass
