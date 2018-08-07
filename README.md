# cape-machine-reader [![CircleCI](https://circleci.com/gh/bloomsburyai/cape-machine-reader.svg?style=svg&circle-token=62818f17fe9047851372b7ba8fa0037a2593eebe)](https://circleci.com/gh/bloomsburyai/cape-machine-reader)

Common interface for cape machine readers.
More detailed tutorial/documentation can be found [Here](docs.md)


## Who is this repo for?

For low-level access to a Question answering functionality the [MachineReader objects](https://github.com/bloomsburyai/cape-machine-reader/blob/master/cape_machine_reader/cape_machine_reader_core.py) object can be used, but most users may find the functionality exposed by [cape-responder](https://github.com/bloomsburyai/cape-responder) to be more complete and easy to
work with at a higher level.
This repo is also helpful for those wishing to build and integrate their own machine reading models into Cape.
To integrate a new machine reading model, implement the interface of [cape_machine_reader_model.py](https://github.com/bloomsburyai/cape-machine-reader/blob/master/cape_machine_reader/cape_machine_reader_model.py)


## Who is this repo not for?

Most users may find the functionality exposed by [cape-responder](https://github.com/bloomsburyai/cape-responder) to be more complete and easy to
work with at a higher level.