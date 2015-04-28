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

public class MatrixFactCore {
    private static final Logger logger =
        LoggerFactory.getLogger(MatrixFactCore.class);

    // Perform a single SGD on a rating and update LTable and RTable
    // accordingly.
    public static void sgdOneRating(Rating r, double learningRate,
            DoubleTable LTable, DoubleTable RTable, int K, double lambda) {
    	//get Li and Rj
    	DoubleRow Li = new DenseDoubleRow(K+1);
    	DoubleRow iRow = LTable.get(r.userId);
    	Li.reset(iRow);
    	DoubleRow Rj = new DenseDoubleRow(K+1);
    	DoubleRow jRow = RTable.get(r.prodId);
    	Rj.reset(jRow);
    	//set ni,mj
    	double ni = lambda/Li.getUnlocked(K);
    	double mj = lambda/Rj.getUnlocked(K);
    	double eij = (double)r.rating - product(Li,Rj,K);
    	//update Li and Rj
    	DoubleRowUpdate up_Li = learn(Li,Rj,learningRate,eij,ni,K);
    	DoubleRowUpdate up_Rj = learn(Rj,Li,learningRate,eij,mj,K);
    	//put updated Li and Rj back to tables?? update means covering?
    	LTable.batchInc(r.userId, up_Li);
    	LTable.batchInc(r.prodId, up_Rj);
        // TODO
    }
    public static DoubleRowUpdate learn(DoubleRow Li,DoubleRow Rj, double learningRate,double eij,double ni,int K){
    	DoubleRowUpdate up_Li = new DenseDoubleRowUpdate(K+1);
    	for(int i = 0; i < K; i++){
    		double li = Li.getUnlocked(i);
    		double rj = Rj.getUnlocked(i);
    		double new_li = li + 2*learningRate*(eij*rj - ni*li);
    		up_Li.setUpdate(i, new_li);
    	}
    	up_Li.setUpdate(K, ni);
    	return up_Li;
    }
    public static double product(DoubleRow Li, DoubleRow Rj, int K){
    	double result = 0.0;
    	for(int i = 0; i < K; i++){
    		result += Li.getUnlocked(i)*Rj.getUnlocked(i);
    	}
    	return result;
    }
    // Evaluate square loss on entries [elemBegin, elemEnd), and L2-loss on of
    // row [LRowBegin, LRowEnd) of LTable,  [RRowBegin, RRowEnd) of Rtable.
    // Note the interval does not include LRowEnd and RRowEnd. Record the loss to
    // lossRecorder.
    public static void evaluateLoss(ArrayList<Rating> ratings, int ithEval,
            int elemBegin, int elemEnd, DoubleTable LTable,
            DoubleTable RTable, int LRowBegin, int LRowEnd, int RRowBegin,
            int RRowEnd, LossRecorder lossRecorder, int K, double lambda) {
    	//no idea what elembegin and elemend is...
        // TODO
    	//initalization
        double sqLoss = 0;
        double totalLoss = 0;
        HashSet<Integer> rreg = new HashSet<Integer>();
        HashSet<Integer> lreg = new HashSet<Integer>();
        for(int pointer = elemBegin; pointer < elemEnd; pointer++){
        	//get rating
        	int i = ratings.get(pointer).userId;
        	int j = ratings.get(pointer).prodId;
        	double rating = (double)ratings.get(pointer).rating;
        	
        	//get Li and Rj
        	DoubleRow Li = new DenseDoubleRow(K+1);
        	DoubleRow iRow = LTable.get(i);
        	Li.reset(iRow);
        	DoubleRow Rj = new DenseDoubleRow(K+1);
        	DoubleRow jRow = RTable.get(j);
        	Rj.reset(jRow);
        	
        	//compute sqLoss and totalloss
        	sqLoss += Math.pow(rating-product(Li,Rj,K),2);
        	if(!lreg.contains(i)){
        		totalLoss += product(Li,Li,K);
        		lreg.add(i);
        	}
        	if(!rreg.contains(j)){
        		totalLoss += product(Rj,Rj,K);
        		rreg.add(j);
        	}
        }
        totalLoss *= lambda;
        totalLoss += sqLoss;
        lossRecorder.incLoss(ithEval, "SquareLoss", sqLoss);
        lossRecorder.incLoss(ithEval, "FullLoss", totalLoss);
        lossRecorder.incLoss(ithEval, "NumSamples", elemEnd - elemBegin);
    }
}
