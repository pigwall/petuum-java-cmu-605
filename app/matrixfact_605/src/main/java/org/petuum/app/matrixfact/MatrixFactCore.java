package org.petuum.app.matrixfact;

import org.petuum.app.matrixfact.Rating;
import org.petuum.app.matrixfact.LossRecorder;
import org.petuum.ps.PsTableGroup;
import org.petuum.ps.row.double_.DenseDoubleRow;
import org.petuum.ps.row.double_.DenseDoubleRowUpdate;
import org.petuum.ps.row.double_.DoubleRow;
import org.petuum.ps.row.double_.DoubleRowUpdate;
import org.petuum.ps.table.DoubleTable;
import org.petuum.ps.common.util.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class MatrixFactCore {
    private static final Logger logger =
        LoggerFactory.getLogger(MatrixFactCore.class);

    // Perform a single SGD on a rating and update LTable and RTable
    // accordingly.
    public static void sgdOneRating(Rating r, double learningRate,
            DoubleTable LTable, DoubleTable RTable, int K, double lambda) {
        
        // Calculate the gradient
        DoubleRow rowCacheL = new DenseDoubleRow(K+1);
        DoubleRow LRow = LTable.get(r.userId);
        rowCacheL.reset(LRow);
        double n_i = rowCacheL.getUnlocked(K);
        
        DoubleRow rowCacheR = new DenseDoubleRow(K+1);
        DoubleRow RRow = RTable.get(r.prodId);
        rowCacheR.reset(RRow);
        double m_j = rowCacheR.getUnlocked(K);
        
        // calculate e_ij
        double e_ij = r.rating;
        for (int k = 0; k < K; ++k) {
                e_ij -= rowCacheL.getUnlocked(k) * rowCacheR.getUnlocked(k);
        }
        
        // update L table
        DoubleRowUpdate updates = new DenseDoubleRowUpdate(K+1);
        for (int k = 0; k < K; ++k) {
            double delta = e_ij * rowCacheR.getUnlocked(k) - lambda / n_i
                    * rowCacheL.getUnlocked(k);
            updates.setUpdate(k, 2 * learningRate * delta);
        }
        LTable.batchInc(r.userId, updates);
        
        
        // update R table
        updates = new DenseDoubleRowUpdate(K+1);
        for (int k = 0; k < K; ++k) {
            double delta = e_ij * rowCacheL.getUnlocked(k) - lambda / m_j
                    * rowCacheR.getUnlocked(k);
            updates.setUpdate(k, 2 * learningRate * delta);
        }
        RTable.batchInc(r.prodId, updates);        
    }

    // Evaluate square loss on entries [elemBegin, elemEnd), and L2-loss on of
    // row [LRowBegin, LRowEnd) of LTable,  [RRowBegin, RRowEnd) of Rtable.
    // Note the interval does not include LRowEnd and RRowEnd. Record the loss to
    // lossRecorder.
    public static void evaluateLoss(ArrayList<Rating> ratings, int ithEval,
            int elemBegin, int elemEnd, DoubleTable LTable,
            DoubleTable RTable, int LRowBegin, int LRowEnd, int RRowBegin,
            int RRowEnd, LossRecorder lossRecorder, int K, double lambda) {

        double sqLoss = 0;
        double totalLoss = 0;
        Set<Integer> LSet = new HashSet<Integer>();
        Set<Integer> RSet = new HashSet<Integer>();
        
        for (int i = elemBegin; i < elemEnd; ++i) {
            Rating r = ratings.get(i);
            
            DoubleRow rowCacheL = new DenseDoubleRow(K+1);
            DoubleRow LRow = LTable.get(r.userId);
            rowCacheL.reset(LRow);
            DoubleRow rowCacheR = new DenseDoubleRow(K+1);
            DoubleRow RRow = RTable.get(r.prodId);
            rowCacheR.reset(RRow);
            
            // calculate estimated rating
            double rating = 0.0; 
            for (int k = 0; k < K; ++k) {
                rating += rowCacheL.getUnlocked(k) * rowCacheR.getUnlocked(k);
            }
            double diff = r.rating - rating;
            sqLoss += diff * diff;
            
            // calculate regularization err
            if (r.userId >= LRowBegin && r.userId < LRowEnd && !LSet.contains(r.userId)) {
                LSet.add(r.userId);
                double loss = 0.0;
                for (int k = 0; k < K; ++k) {
                    double w = rowCacheL.getUnlocked(k);
                    loss += w * w;
                }
                totalLoss += loss;
            }
            
            if (r.prodId >= RRowBegin && r.prodId < RRowEnd && !RSet.contains(r.prodId)) {
                RSet.add(r.prodId);
                double loss = 0.0;
                for (int k = 0; k < K; ++k) {
                    double w = rowCacheR.getUnlocked(k);
                    loss += w * w;
                }
                totalLoss += loss;
            }
        }
        
        // regularization err
        for (int row = LRowBegin; row < LRowEnd; ++row) {
            if (LSet.contains(row)) {
                continue;
            }
            
            DoubleRow rowCacheL = new DenseDoubleRow(K+1);
            DoubleRow LRow = LTable.get(row);
            rowCacheL.reset(LRow);
            
            double loss = 0.0;
            for (int k = 0; k < K; ++k) {
                double w = rowCacheL.getUnlocked(k);
                loss += w * w;
            }
            totalLoss += loss;
        }
        
        for (int col = RRowBegin; col < RRowEnd; ++col) {
            if (RSet.contains(col)) {
                continue;
            }
            
            DoubleRow rowCacheR = new DenseDoubleRow(K+1);
            DoubleRow RRow = RTable.get(col);
            rowCacheR.reset(RRow);
            
            double loss = 0.0;
            for (int k = 0; k < K; ++k) {
                double w = rowCacheR.getUnlocked(k);
                loss += w * w;
            }
            totalLoss += loss;
        }
        
        // apply lambda
        totalLoss = totalLoss * lambda + sqLoss;        
        
        lossRecorder.incLoss(ithEval, "SquareLoss", sqLoss);
        lossRecorder.incLoss(ithEval, "FullLoss", totalLoss);
        lossRecorder.incLoss(ithEval, "NumSamples", elemEnd - elemBegin);
    }
}
