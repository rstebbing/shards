##########################################
# File: pickle_.py                       #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import cPickle

# dump
def dump(path, obj, raise_on_failure=True):
    own_fid = False
    if isinstance(path, basestring):
        path = open(path,'w+b')
        own_fid = True

    try:
        cPickle.dump(obj, path, cPickle.HIGHEST_PROTOCOL)
    except IOError:
        if raise_on_failure:
            raise
    finally:
        if own_fid:
            try:
                path.close()
            except IOError:
                pass

