import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import skimage.transform
import cPickle as pickle
import copy
import json
from tqdm import tqdm

from utils.nn import NN
from utils.coco.coco import COCO
from utils.coco.pycocoevalcap.eval import COCOEvalCap
from utils.misc import ImageLoader, CaptionData, TopN

import IPython


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        self.image_shape = [224, 224, 3]
        self.nn = NN(config)
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)
        self.build()

    def build(self):
        raise NotImplementedError()

    def train(self, sess, train_data):
        """ Train the model using the COCO train2014 data. """
        print("Training the model...")
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)

        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()
                image_files, sentences, masks = batch
                images = self.image_loader.load_images(image_files)
                feed_dict = {self.images: images,
                             self.sentences: sentences,
                             self.masks: masks}
                _, summary, global_step = sess.run([self.opt_op,
                                                    self.summary,
                                                    self.global_step],
                                                    feed_dict=feed_dict)
                if (global_step + 1) % config.save_period == 0:
                    self.save()
                train_writer.add_summary(summary, global_step)
            train_data.reset()

        self.save()
        train_writer.close()
        print("Training complete.")

    def eval(self, sess, eval_gt_coco, eval_data, vocabulary):
        """ Evaluate the model using the COCO val2014 data. """
        print("Evaluating the model ...")
        config = self.config

        results = []
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

        # Generate the captions for the images
        idx = 0
        for k in tqdm(list(range(eval_data.num_batches)), desc='batch'):
            batch = eval_data.next_batch()
            caption_data = self.beam_search(sess, batch, vocabulary)

            fake_cnt = 0 if k<eval_data.num_batches-1 \
                         else eval_data.fake_count
            for l in range(eval_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                results.append({'image_id': eval_data.image_ids[idx],
                                'caption': caption})
                idx += 1

                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = plt.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.eval_result_dir,
                                             image_name+'_result.jpg'))

        fp = open(config.eval_result_file, 'wb')
        json.dump(results, fp)
        fp.close()

        # Evaluate these captions
        eval_result_coco = eval_gt_coco.loadRes(config.eval_result_file)
        scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)
        scorer.evaluate()
        print("Evaluation complete.")

    def _test_save_attention_sequence(self, image_file_path, alpha_seq, word_seq, test_result_dir, verbose=True):
        assert(len(alpha_seq) == len(word_seq))

        # parse the image file name and extension
        image_file_name = image_file_path.split(os.sep)[-1]
        image_name, image_ext = os.path.splitext(image_file_name)

        # verbose output
        if verbose:
            print(word_seq)

        # construct attended images
        seg = (14, 14) # 196 = 14 * 14 (last conv layer of VGG16)
        attention_scale_base = 0.3
        img = cv2.imread(image_file_path)
        for i in range(len(alpha_seq)):
            attention = np.reshape(alpha_seq[i], seg)
            word = word_seq[i]

            # rescale attention
            attention_scale_min = np.amin(attention)
            attention_scale_max = np.amax(attention)
            attention_scaled = (attention / attention_scale_max) * (1.0 - attention_scale_base) + attention_scale_base

            # stack attention
            attention_stacked = np.stack([attention_scaled, attention_scaled, attention_scaled], 2)

            # construct mask
            mask = skimage.transform.pyramid_expand(attention_stacked, upscale=16, sigma=20) # 14 * 16 = 224
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

            # construct weighted image
            img_weighted = np.uint8(img * mask)

            # save weighted image
            #TODO

            # verbose output
            if True:
                print('[ %s ] scale: %f ~ %f' % (word, attention_scale_min, attention_scale_max))
                cv2.imshow('img_weighted', img_weighted)
                cv2.waitKey(0)

    def _test_save_result(self, image_file_path, caption, test_result_dir):
        image_file_name = image_file_path.split(os.sep)[-1]
        image_name, image_ext = os.path.splitext(image_file_name)
        img = plt.imread(image_file_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(caption)
        plt.savefig(os.path.join(test_result_dir, image_name + '_result' + image_ext))

    def test(self, sess, test_data, vocabulary):
        """ Test the model using any given images. """
        print("Testing the model ...")
        config = self.config

        if not os.path.exists(config.test_result_dir):
            os.mkdir(config.test_result_dir)

        captions = []
        scores = []

        # Generate the captions for the images
        for k in tqdm(list(range(test_data.num_batches)), desc='path'):
            batch = test_data.next_batch()
            caption_data = self.beam_search(sess, batch, vocabulary)

            fake_cnt = 0 if k<test_data.num_batches-1 \
                         else test_data.fake_count
            for l in range(test_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                alpha_seq = caption_data[l][0].alpha_seq
                caption = vocabulary.get_sentence(word_idxs)
                captions.append(caption)
                scores.append(score)

                # Save word attentions in image files
                if True:
                    image_file_path = batch[l]
                    word_seq = []
                    for i_word in range(len(word_idxs)):
                        word_idx = word_idxs[i_word]
                        word = vocabulary.words[word_idx]
                        word_seq.append(word)
                    self._test_save_attention_sequence(image_file_path, alpha_seq, word_seq, config.test_result_dir)

                # Save the result in an image file
                if False:
                    image_file_path = batch[l]
                    self._test_save_result(image_file_path, caption, config.test_result_dir)

        # Save the captions to a file
        results = pd.DataFrame({'image_files':test_data.image_files,
                                'caption':captions,
                                'prob':scores})
        results.to_csv(config.test_result_file)
        print("Testing complete.")

    def beam_search(self, sess, image_files, vocabulary):
        """Use beam search to generate the captions for a batch of images."""
        # Feed in the images to get the contexts and the initial LSTM states
        config = self.config
        images = self.image_loader.load_images(image_files)
        contexts, initial_memory, initial_output = sess.run(
            [self.conv_feats, self.initial_memory, self.initial_output],
            feed_dict = {self.images: images})

        partial_caption_data = []
        complete_caption_data = []
        for k in range(config.batch_size):
            initial_beam = CaptionData(sentence = [],
                                       memory = initial_memory[k],
                                       output = initial_output[k],
                                       score = 1.0,
                                       alpha_seq = [])
            partial_caption_data.append(TopN(config.beam_size))
            partial_caption_data[-1].push(initial_beam)
            complete_caption_data.append(TopN(config.beam_size))

        # Run beam search
        for idx in range(config.max_caption_length):
            partial_caption_data_lists = []
            for k in range(config.batch_size):
                data = partial_caption_data[k].extract()
                partial_caption_data_lists.append(data)
                partial_caption_data[k].reset()

            num_steps = 1 if idx == 0 else config.beam_size
            for b in range(num_steps):
                if idx == 0:
                    last_word = np.zeros((config.batch_size), np.int32)
                else:
                    last_word = np.array([pcl[b].sentence[-1]
                                        for pcl in partial_caption_data_lists],
                                        np.int32)

                last_memory = np.array([pcl[b].memory
                                        for pcl in partial_caption_data_lists],
                                        np.float32)
                last_output = np.array([pcl[b].output
                                        for pcl in partial_caption_data_lists],
                                        np.float32)

                memory, output, scores, alpha = sess.run(
                    [self.memory, self.output, self.probs, self.alpha],
                    feed_dict = {self.contexts: contexts,
                                 self.last_word: last_word,
                                 self.last_memory: last_memory,
                                 self.last_output: last_output})

                # Find the beam_size most probable next words
                for k in range(config.batch_size):
                    caption_data = partial_caption_data_lists[k][b]
                    words_and_scores = list(enumerate(scores[k]))
                    words_and_scores.sort(key=lambda x: -x[1])
                    words_and_scores = words_and_scores[0:config.beam_size+1]

                    # Append each of these words to the current partial caption
                    for w, s in words_and_scores:
                        sentence = caption_data.sentence + [w]
                        alpha_seq = caption_data.alpha_seq + [alpha[k]]
                        score = caption_data.score * s
                        beam = CaptionData(sentence,
                                           memory[k],
                                           output[k],
                                           score,
                                           alpha_seq=alpha_seq)
                        if vocabulary.words[w] == '.':
                            complete_caption_data[k].push(beam)
                        else:
                            partial_caption_data[k].push(beam)

        results = []
        for k in range(config.batch_size):
            if complete_caption_data[k].size() == 0:
                complete_caption_data[k] = partial_caption_data[k]
            results.append(complete_caption_data[k].extract(sort=True))

        return results

    def save(self):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path+".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path).item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse = True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                    except ValueError:
                        pass
        print("%d tensors loaded." %count)
