/*
 * CUDARewriter.cpp
 *
 *  Created on: 24/jul/15
 *      Author: latzori
 */

#include <string>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <iterator>
#include <array>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"


#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm::sys::path;

//TODO Is this mandatory?
static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");

/*
 * This class takes care of augmenting the dimensionality of the local CUDA thread variables
 * and of the movement of the declarations in the right places
 *
 */
//
//
class dimensionality_augmenter : public ASTConsumer {
public:
	dimensionality_augmenter(CompilerInstance *comp, Rewriter * R) : ASTConsumer(), CI(comp), Rew(R) { }
	virtual ~dimensionality_augmenter() { }
	virtual void Initialize(ASTContext &Context) {
		SM = &Context.getSourceManager();
		LO = &CI->getLangOpts();
		PP = &CI->getPreprocessor();
	}

	virtual bool HandleTopLevelDecl(DeclGroupRef DG) {

		Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
		SourceLocation sloc = SM->getSpellingLoc(firstDecl->getLocation());

		/*  Checking which file we are scanning.
		 *  We skip everything apart from the main file.
		 *  FIXME: "extern" keyword in #defines could cause problems
		 */
		if( SM->getFileID(sloc) != SM->getMainFileID()){ //FileID mismatch
			return true; //Just skip the loc but continue parsing.
		}

        //Walk and rewrite declarations in group
        for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
            //Handles globally defined functions
            if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*i)) {
                    if (fd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>()) {
                    	//FIXME Also CUDA Device has to be taken into account? Future work about functions called inside the kernels?
                    	//Device function, so rewrite kernel
                        RewriteKernelFunction(fd);
                    }
            }
        }
        return true;
	}

private:
    CompilerInstance *CI;
    SourceManager *SM;
    LangOptions *LO;
    Preprocessor *PP;
    Rewriter *Rew;
    std::set<std::string> KernelDecls; //Just a set to see if we find DeclRefExpr matching the declarations we found before
    std::vector<std::string> NewDecls; //Vector to save the declarations temporarily
    SourceLocation kernelbodystart; //The start location of the kernel of the function

    /*
     * String representing the beginning of a thread loop //TODO not needed anymore in the dimensionality_augmenter?
     */
    std::string TL_START1 = "for(threadIdx.z=0; threadIdx.z < blockDim.z; threadIdx.z++){\n";
    std::string TL_START2 = "for(threadIdx.y=0; threadIdx.y < blockDim.y; threadIdx.y++){\n";
    std::string TL_START3 = "for(threadIdx.x=0; threadIdx.x < blockDim.x; threadIdx.x++){\n";
    std::string tid = "tid=threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y;"; //TODO MACRO for tid instead of recalculating it everytime? (it's the same maybe)
    std::string TL_START = TL_START1+TL_START2+TL_START3+tid;


    //String representing the end of a thread loop //TODO not needed anymore in the dimensionality_augmenter?
    std::string TL_END = "}}}";
    /*
     * Private method to prepare the function rewriting
     * FIXME 1 At the moment the only scope we handle is the global one of the function.
     * FIXME 1 We need to take into account multiple declarations of variables with the same names in different scopes,
     * FIXME 1 so moving everything at the top of the scope on which we declared the variable and not at the beginning of the body.
     *
     * FIXME 2 We need to capture the value of the block dimension in the kernel call in the host code if we want to augment the dimensionality
     * FIXME 2 in a static way. Otherwise we need to allocate the new variables dinamically. FIXME!Now the dimension is just hardcoded!
     */
    void RewriteKernelFunction(FunctionDecl* kf) {

	   if (Stmt *body = kf->getBody()){
		kernelbodystart = PP->getLocForEndOfToken(body->getLocStart());
		replicate(body); //Call to the analysis of the variables

		//We have also to declare the now explicit threadIdx, the tid to access the variables, and the dimension of the new dimensionality
		std::string initsupport = "\ndim3 threadIdx;\nint tid;\nint numThreads = 32;\n";
		for(int i = 0; i < NewDecls.size(); i++){
			//We add to initsupport the list of the declarations we are moving
			initsupport += NewDecls[i] + "\n";
		}
		//And then we insert everything at the beginning of the body.
		Rew->InsertTextAfter(kernelbodystart, initsupport);
	   } //else {} // Empty kernel (possible scenarios?)
    }


    /*
     * Private method that actually executes the augmenting of the dimensionality.
     *
     * FIXME 1 We are not keeping the information about the actual scope. This can be done casting before to compound statements?
     *
     * FIXME 2 What about the __constant__ variables?
     */
    void replicate(Stmt *s){

    		//Casting to compoundstmt
        	//if(CompoundStmt *cs = dyn_cast<CompoundStmt>(s)){
        	//	for(Stmt::child_iterator i = cs->body_begin(), e = cs->body_end(); i!=e; ++i){
        	//		if(*i){
    		//declaration inside a compound statement, which is a new scope

    		//For every declaration we find we save them in a vector.
    		if (DeclStmt *ds = dyn_cast<DeclStmt>(s)){
    			DeclGroupRef DG = ds->getDeclGroup();
    			for (DeclGroupRef::iterator i2 = DG.begin(), e = DG.end(); i2!=e; ++i2){
    				if(*i2){ //not null
    					if(VarDecl *vd = dyn_cast<VarDecl>(*i2)){ //Found a VarDecl
    					   /*
    						* This declaration is tagged as shared, so we don't have to augment its dimensionality.
    						* However is still needed to move it to the top of the scope.
    						* FIXME 1 The dimension of the array is calculated before the translation if #defined
    						* FIXME 1 and written in a strange way if we get the type. TODO ConstantArray?
    						* FIXME 2 For some reason we're not able to remove the declaration from the original location.4
    						*/
    						if (CUDASharedAttr *sharedAttr = vd->getAttr<CUDASharedAttr>()) {
    							//Workaround for bugged gettype, we take all the text.
    							//SourceRange a = SourceRange(PP->getLocForEndOfToken(vd->getLocStart()), vd->getLocEnd());
    							//StringRef test = Lexer::getSourceText(CharSourceRange(a, false), *SM, *LO);
    							//NewDecls.push_back(test.str()+";");

    							//Bugged getType
    							//NewDecls.push_back(vd->getType().getAsString()+ " "+ vd->getNameAsString()+";");

    							//If it has an initialization in the same statement of the declaration, we have to keep it.
        						//if(vd->hasInit()){
        							//Rew->ReplaceText(SourceRange((*i2)->getLocStart(), PP->getLocForEndOfToken((*i2)->getLocEnd())), vd->getNameAsString() + " = " + getStmtText(vd->getInit()) + ";");
       						    //} else {
    								//No initialization, we just move it and delete the old string.
    								//FIXME BUGGED DELETING
        							//StringRef test2 = Lexer::getSourceText(CharSourceRange(SourceRange((*i2)->getLocStart(), PP->getLocForEndOfToken((*i2)->getLocEnd())), true), *SM, *LO);
       							    //Rew->RemoveText(SourceRange((*i2)->getLocStart(), PP->getLocForEndOfToken((*i2)->getLocEnd())));
        							//Rew->ReplaceText(SourceRange((*i2)->getLocStart(), PP->getLocForEndOfToken((*i2)->getLocEnd())), "");
        						//}
    						} else {//Not a shared variable, we also have to augment the dimensionality!
    							//Inserting the variables in a set just for the match with the refs
    							KernelDecls.insert(vd->getNameAsString());
    							//Inserting the declaration string in the vector, adding the new dimensionality
    							//FIXME BlockDimension also here?
								NewDecls.push_back(vd->getType().getAsString()+" "+vd->getNameAsString()+"[numThreads];");
								//Again if there is an initialization, we have to keep it.
								if(vd->hasInit()){
									Rew->ReplaceText(SourceRange(vd->getLocStart(), PP->getLocForEndOfToken(vd->getLocEnd())), vd->getNameAsString() + "[tid] = " + getStmtText(vd->getInit()) + ";");
								} else { //otherwise we delete everything.
									Rew->ReplaceText(SourceRange(vd->getLocStart(), PP->getLocForEndOfToken(vd->getLocEnd())), "");
								}
    						}

    					}
    				}
    			}
    		}
    		else if(DeclRefExpr *dre = dyn_cast<DeclRefExpr>(s)){ //Searching for DeclRefExpr
    			if(KernelDecls.find(dre->getNameInfo().getAsString()) != KernelDecls.end()){ //We found one matching the declarations saved before
    				//Then we access it by thread id.
    				Rew->ReplaceText(SourceRange(dre->getLocStart(), dre->getLocEnd()), dre->getNameInfo().getAsString()+"[tid]");
    			}
    		}
    		//Keep iterating
    		for (Stmt::child_iterator s_ci = s->child_begin(), s_ce = s->child_end(); s_ci != s_ce; ++s_ci) {
    			if(*s_ci){
    				replicate(*s_ci);
    			}

    		}
    }

    //Utility to get the text from a statement
    //Technique found in the CU2CL implementation.
    std::string getStmtText(Stmt *s) {
        SourceLocation start(SM->getExpansionLoc(s->getLocStart())), end(Lexer::getLocForEndOfToken(SourceLocation(SM->getExpansionLoc(s->getLocEnd())), 0,  *SM, *LO));
        return std::string(SM->getCharacterData(start), SM->getCharacterData(end)-SM->getCharacterData(start));
    }

};

//Our ASTFrontendAction.
class Replication : public ASTFrontendAction{
public:
	Replication(){}
	void EndSourceFileAction() override {
		//For now we just use llvm::outs, which is a raw_ostream referencing the standard output. Then we redirect it to a temporary file
		// or to the input of the Rewriter tool.
		//TODO can we just make them communicate together via source code? (Directly or with a temporary file?)
		TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());

	}

	std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
		TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
		return llvm::make_unique<dimensionality_augmenter>(&CI, &TheRewriter);
	}
private:
	Rewriter TheRewriter;
};


int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());
  return Tool.run(newFrontendActionFactory<Replication>().get());
}
