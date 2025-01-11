## l0_attack.py + l2_attack.py + li_attack.py-- attack a network optimizing for l_0, l_2 or l_infinity distance
## This is just a copy and paste from https://github.com/carlini/nn_robust_attacks.
## TODO: merge the code?
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from util import lid_adv_term
# Settings for C&W L2 attack
L2_BINARY_SEARCH_STEPS = 9  # Number of times to adjust the constant with binary search
L2_MAX_ITERATIONS = 1000    # Number of iterations for optimization
L2_LEARNING_RATE = 1e-2     # Learning rate
L2_TARGETED = True          # Whether to target a specific class
L2_CONFIDENCE = 0           # Confidence level of adversarial examples
L2_INITIAL_CONST = 1e-3     # Initial tradeoff constant

class CarliniL2:
    def __init__(self, model, image_size, num_channels, num_labels, batch_size=100,
                 confidence=L2_CONFIDENCE, targeted=L2_TARGETED, learning_rate=L2_LEARNING_RATE,
                 binary_search_steps=L2_BINARY_SEARCH_STEPS, max_iterations=L2_MAX_ITERATIONS,
                 abort_early=True, initial_const=L2_INITIAL_CONST, device='cuda'):
        """
        The L2 optimized attack in PyTorch.
        """
        self.model = model
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.confidence = confidence
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def attack(self, inputs, labels):
        """
        Perform the attack on the input images and return adversarial examples.
        """
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Initialize variables
        modifier = torch.zeros_like(inputs, requires_grad=True, device=self.device)
        optimizer = optim.Adam([modifier], lr=self.learning_rate)

        best_adv = inputs.clone()
        best_loss = torch.full((self.batch_size,), float('inf'), device=self.device)

        for step in range(self.max_iterations):
            adv_images = torch.tanh(modifier + torch.atanh(inputs * 2 - 1)) / 2
            outputs = self.model(adv_images)

            # Compute loss
            real = torch.sum(labels * outputs, dim=1)
            other = torch.max((1 - labels) * outputs - labels * 1e4, dim=1)[0]

            if self.targeted:
                loss1 = torch.clamp(other - real + self.confidence, min=0)
            else:
                loss1 = torch.clamp(real - other + self.confidence, min=0)

            l2dist = torch.sum((adv_images - inputs) ** 2, dim=(1, 2, 3))
            loss = torch.sum(self.initial_const * loss1 + l2dist)

            # Update modifier
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Early stopping
            if step % 10 == 0 or step == self.max_iterations - 1:
                with torch.no_grad():
                    preds = outputs.argmax(dim=1)
                    if self.targeted:
                        successful = (preds == labels.argmax(dim=1))
                    else:
                        successful = (preds != labels.argmax(dim=1))

                    improved = successful & (l2dist < best_loss)
                    best_adv[improved] = adv_images[improved]
                    best_loss[improved] = l2dist[improved]

                    if self.abort_early and step > self.max_iterations // 10:
                        if not improved.any():
                            break

        return best_adv
    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x, y):
            x = x.clone().detach()
            x[y] -= self.confidence
            x = x.argmax()
            if self.targeted:
                return x == y
            else:
                return x != y

        batch_size = imgs.size(0)

        # Convert to tanh-space
        imgs_tanh = torch.atanh(imgs * 1.999999).to(self.device)
        labs = labs.to(self.device)

        # Initialize bounds
        lower_bound = torch.zeros(batch_size, device=self.device)
        const = torch.full((batch_size,), self.initial_const, device=self.device)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)

        # Initialize best results
        o_bestl2 = torch.full((batch_size,), 1e10, device=self.device)
        o_bestscore = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        o_bestattack = imgs.clone().to(self.device)

        for outer_step in range(self.binary_search_steps):
            # Reset optimizer and modifier
            modifier = torch.zeros_like(imgs, requires_grad=True, device=self.device)
            optimizer = torch.optim.Adam([modifier], lr=self.learning_rate)

            bestl2 = torch.full((batch_size,), 1e10, device=self.device)
            bestscore = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

            # Update constant for last binary search step
            if self.repeat and outer_step == self.binary_search_steps - 1:
                const = upper_bound

            prev_loss = torch.full((batch_size,), 1e6, device=self.device)
            for iteration in range(self.max_iterations):
                # Generate adversarial examples
                adv_images = torch.tanh(modifier + imgs_tanh) / 2
                outputs = self.model(adv_images)

                # Compute loss
                real = torch.sum(labs * outputs, dim=1)
                other = torch.max((1 - labs) * outputs - labs * 1e4, dim=1)[0]

                if self.targeted:
                    loss1 = torch.clamp(other - real + self.confidence, min=0)
                else:
                    loss1 = torch.clamp(real - other + self.confidence, min=0)

                l2dist = torch.sum((adv_images - imgs) ** 2, dim=(1, 2, 3))
                loss = torch.sum(const * loss1 + l2dist)

                # Perform gradient update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Early stopping
                if self.abort_early and iteration % (self.max_iterations // 10) == 0:
                    if torch.all(loss > prev_loss * 0.9999):
                        break
                    prev_loss = loss

                # Update best results
                for e in range(batch_size):
                    if l2dist[e] < bestl2[e] and compare(outputs[e], labs[e].argmax()):
                        bestl2[e] = l2dist[e]
                        bestscore[e] = outputs[e].argmax()
                    if l2dist[e] < o_bestl2[e] and compare(outputs[e], labs[e].argmax()):
                        o_bestl2[e] = l2dist[e]
                        o_bestscore[e] = outputs[e].argmax()
                        o_bestattack[e] = adv_images[e]

            # Adjust the constant for binary search
            for e in range(batch_size):
                if compare(bestscore[e], labs[e].argmax()) and bestscore[e] != -1:
                    upper_bound[e] = min(upper_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    lower_bound[e] = max(lower_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        const[e] *= 10

        # Compute success rate
        success_rate = 1 - torch.sum(o_bestl2 == 1e10).item() / batch_size
        print(f'Success rate: {success_rate:.4f}')
        return o_bestattack



class CarliniLID:
    def __init__(self, model, image_size, num_channels, num_labels, batch_size=100,
                 confidence=1e-3, targeted=True, learning_rate=1e-2,
                 binary_search_steps=9, max_iterations=1000,
                 abort_early=True, initial_const=1e-3):

        self.model = model
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        # The variable we're going to optimize over
        self.modifier = torch.zeros((batch_size, num_channels, image_size, image_size), requires_grad=True)

        # These are the variables to send data to the model
        self.timg = torch.zeros((batch_size, num_channels, image_size, image_size))
        self.tlab = torch.zeros((batch_size, num_labels))
        self.const = torch.zeros(batch_size)

        self.optimizer = optim.Adam([self.modifier], lr=self.LEARNING_RATE)

    def attack(self, X, Y):
        """
        Perform the L_2 attack on the given images for the given targets.
        :param X: samples to generate advs
        :param Y: the original class labels
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        nb_classes = Y.shape[1]

        # Random select target class for targeted attack
        y_target = np.copy(Y)
        if self.TARGETED:
            for i in range(Y.shape[0]):
                current = int(np.argmax(Y[i]))
                target = np.random.choice([i for i in range(nb_classes) if i != current])
                y_target[i] = np.eye(nb_classes)[target]

        X_adv = np.zeros_like(X)
        for i in tqdm(range(0, X.shape[0], self.batch_size)):
            start = i
            end = i + self.batch_size
            end = np.minimum(end, X.shape[0])
            X_adv[start:end] = self.attack_batch(X[start:end], y_target[start:end])

        return X_adv

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """

        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x[y] -= self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = imgs.shape[0]

        # Convert to tanh-space
        imgs = np.arctanh(imgs * 1.999999)

        # Set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # The best l2, score, and image attack
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            self.optimizer.zero_grad()
            batch = torch.tensor(imgs[:batch_size], requires_grad=True)
            batchlab = torch.tensor(labs[:batch_size])

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size

            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # Forward pass to get clean logits (before attack)
            self.model.eval()
            c_logits = self.model(batch)

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                self.optimizer.zero_grad()

                # Perform the attack
                output = self.model(batch + self.modifier)
                l2dist = torch.sum((batch + self.modifier - batch)**2, dim=[1, 2, 3])
                real = torch.sum((batchlab) * output, 1)
                other = torch.max((1 - batchlab) * output - (batchlab * 10000), 1)

                if self.TARGETED:
                    loss1 = torch.maximum(torch.zeros_like(other), other - real + self.CONFIDENCE)
                else:
                    loss1 = torch.maximum(torch.zeros_like(real), real - other + self.CONFIDENCE)

                loss_lid = self.lid_adv_term(c_logits, output)
                loss2 = torch.sum(l2dist)
                loss1 = torch.sum(CONST * (loss1 + loss_lid))

                loss = loss1 + loss2

                loss.backward()
                self.optimizer.step()

                for e, (l2, sc, ii) in enumerate(zip(l2dist, output, batch)):
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # Adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10

        o_bestl2 = np.array(o_bestl2)
        print('success rate: %.4f' % (1 - np.sum(o_bestl2 == 1e10) / self.batch_size))
        return o_bestattack
