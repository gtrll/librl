from abc import ABC, abstractmethod


class Algorithm(ABC):

    @abstractmethod
    def pretrain(self, gen_ro):
        """ Pretraining. """

    @abstractmethod
    def update(self, ro):
        """ Update the policy based on rollouts. """

    @abstractmethod
    def pi(self, ob, t, done):
        """ Target policy for online querying. """

    @abstractmethod
    def pi_ro(self, ob, t, done):
        """ Behavior policy for online querying. """

    @abstractmethod
    def logp(self, obs, acs):
        """ Log probability of the behavior policy.
            Need to support batch querying. """
