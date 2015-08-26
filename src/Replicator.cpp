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

//This class takes care of the universal replication, and of the movement of the declarations in the right places //fixme don't add thread_loops here? just the first?
class MyReplicator : public ASTConsumer {
public:
	MyReplicator(CompilerInstance *comp, Rewriter * R) : ASTConsumer(), CI(comp), Rew(R) { }
	virtual ~MyReplicator() { }
	virtual void Initialize(ASTContext &Context) {
		SM = &Context.getSourceManager();
		LO = &CI->getLangOpts();
		PP = &CI->getPreprocessor();
	}

	virtual bool HandleTopLevelDecl(DeclGroupRef DG) {


		Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
		//TODO what's the difference between SpellingLoc and Loc?
		//SourceLocation loc = firstDecl->getLocation();
		SourceLocation sloc = SM->getSpellingLoc(firstDecl->getLocation());

		/*  Checking which file we are scanning.
		 *  We skip everything apart from the main file.
		 *  TODO: Rewriting some includes?
		 *  FIXME: "extern" keyword in #defines could cause problems
		 */
		if( SM->getFileID(sloc) != SM->getMainFileID()){ //FileID mismatch
			return true; //Just skip the loc but continue parsing. TODO: Can I skip the whole file?
		}

		//DEBUG std::cout << "HandleTopLevelDecl: " << sloc.printToString(*SM) << "\n";

        //Walk declarations in group and rewrite
        for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
            //Handles globally defined functions
            if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*i)) {
            	//DEBUG std::cout << "\tFunctionDecl2 fd\n";
            	//TODO: Don't translate explicit template specializations
            	//fixme Ignoring templates for now
            	//if(fd->getTemplatedKind() == clang::FunctionDecl::TK_NonTemplate || fd->getTemplatedKind() == FunctionDecl::TK_FunctionTemplate) {
                    if (fd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>()) { //FIXME need device?
                    	//DEBUG std::cout << "\t\tfd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>() = true\n";
                    	//Device function, so rewrite kernel
                        RewriteKernelFunction(fd);
                    }
                  //Templates again.
//                } else {
//                    if (fd->getTemplateSpecializationInfo())
//                    	std::cout << "DEBUG: fd->getTemplateSpecializationInfo = true (Skip?)\n";
//                    else
//                    	std::cout << "DEBUG: Non-rewriteable function without TemplateSpecializationInfo detected?\n";
//                }
            }
            //TODO rewrite type declarations
        }
        return true;
}

private:
    CompilerInstance *CI;
    SourceManager *SM;
    LangOptions *LO;
    Preprocessor *PP;
    Rewriter *Rew;
    std::set<std::string> KernelDecls;
    std::vector<std::string> NewDecls;
    SourceLocation kernelbodystart;

    std::string TL_START1 = "for(threadIdx.z=0; threadIdx.z < blockDim.z; threadIdx.z++){\n";
    std::string TL_START2 = "for(threadIdx.y=0; threadIdx.y < blockDim.y; threadIdx.y++){\n";
    std::string TL_START3 = "for(threadIdx.x=0; threadIdx.x < blockDim.x; threadIdx.x++){\n";
    std::string TL_START = TL_START1+TL_START2+TL_START3+"tid=threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y;";
    //todo macro for tid instead of recalculating it everytime? it's the same maybe


    std::string TL_END = "}}}";

    void RewriteKernelFunction(FunctionDecl* kf) {
           //Parameter Rewriting for a kernel
   //        if (kf->hasBody()) {
           if (Stmt *body = kf->getBody()){
           	//Rew->InsertTextBefore(SM->getExpansionLoc(body->getLocStart()),"prova1\n");
           	//DEBUG std::cout << "\tkf->hasBody() = true\n";
           	//TODO ReorderKernel(body);
           	kernelbodystart = PP->getLocForEndOfToken(body->getLocStart());
           	replicate(body);
           	// tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y; //!!
            std::string a = "\ndim3 threadIdx;\nint tid;\nint numThreads = 32;\n";
            for(int i = 0; i < NewDecls.size(); i++){
               a += NewDecls[i] + "\n";
            }
            //a+= TL_START;
            Rew->InsertTextAfter(kernelbodystart, a);
            //Rew->InsertTextBefore(body->getLocEnd(), TL_END);
           }

           //std::cout << KernelVars.size() << "\n";


    }
    void replicate(Stmt *s){

        	//if(CompoundStmt *cs = dyn_cast<CompoundStmt>(s)){
        	//	for(Stmt::child_iterator i = cs->body_begin(), e = cs->body_end(); i!=e; ++i){
        	//		if(*i){
    		//declaration inside a compound statement, which is a new scope
    		if (DeclStmt *ds = dyn_cast<DeclStmt>(s)){
    			DeclGroupRef DG = ds->getDeclGroup();
    			for (DeclGroupRef::iterator i2 = DG.begin(), e = DG.end(); i2!=e; ++i2){
    			//for(clang::DeclGroupIterator i2 = ds->decl_begin(), e2 = ds->decl_end(); i2 != e2; ++i2){
    				if(*i2){
    					if(VarDecl *vd = dyn_cast<VarDecl>(*i2)){
    						//Decl* a;
    						//std::cout << "here " << vd->getType().getAsString() << vd->getNameAsString() << "\n";
    						if (CUDASharedAttr *sharedAttr = vd->getAttr<CUDASharedAttr>()) {
    							//fixme we don't have to replicate __shared__ memory (BUGGED)
//    						//if(vd->hasAttr<CUDASharedAttr>()){
//    							SourceRange a = SourceRange(PP->getLocForEndOfToken(vd->getLocStart()), vd->getLocEnd());
//
//    							StringRef test = Lexer::getSourceText(CharSourceRange(a, false), *SM, *LO);
//    							std::cout << "Inserting madafakasa: " << test.str() << ";\n";
//    							NewDecls.push_back(test.str()+";");
//    							//NewDecls.push_back(vd->getType().getAsString()+ " "+ vd->getNameAsString()+";");
//        						if(vd->hasInit()){
//        							Rew->ReplaceText(SourceRange((*i2)->getLocStart(), PP->getLocForEndOfToken((*i2)->getLocEnd())), vd->getNameAsString() + " = " + getStmtText(vd->getInit()) + ";");
//        						} else {
//
//        							StringRef test2 = Lexer::getSourceText(CharSourceRange(SourceRange((*i2)->getLocStart(), PP->getLocForEndOfToken((*i2)->getLocEnd())), true), *SM, *LO);
//        							std::cout << "deleting everything madafakka:" << test2.str() << "\n";
//        							//std::cout << "here " << vd->getType(). << vd->getNameAsString() << "\n";
//        							Rew->RemoveText(SourceRange((*i2)->getLocStart(), PP->getLocForEndOfToken((*i2)->getLocEnd())));
//        							//Rew->ReplaceText(SourceRange((*i2)->getLocStart(), PP->getLocForEndOfToken((*i2)->getLocEnd())), "");
//        						}
    						} else {
    							KernelDecls.insert(vd->getNameAsString());
								//std::cout << "inserted vd " << vd->getNameAsString() << "\n";
								NewDecls.push_back(vd->getType().getAsString()+" "+vd->getNameAsString()+"[numThreads];");
								if(vd->hasInit()){
									Rew->ReplaceText(SourceRange(vd->getLocStart(), PP->getLocForEndOfToken(vd->getLocEnd())), vd->getNameAsString() + "[tid] = " + getStmtText(vd->getInit()) + ";");
								} else {
									Rew->ReplaceText(SourceRange(vd->getLocStart(), PP->getLocForEndOfToken(vd->getLocEnd())), "");
								}
    						}

    					}
    				}
    			}
    		}
    		else if(DeclRefExpr *dre = dyn_cast<DeclRefExpr>(s)){
    			//if(dre->getNameInfo().getAsString() isin decls) then "vectorize"

    			//std::cout << "dre " << dre->getNameInfo().getAsString() << "\n";
    			if(KernelDecls.find(dre->getNameInfo().getAsString()) != KernelDecls.end()){
    				//Rew->InsertTextAfter(dre->getLocEnd(), "[tid]");
    				Rew->ReplaceText(SourceRange(dre->getLocStart(), dre->getLocEnd()), dre->getNameInfo().getAsString()+"[tid]");
    				//std::cout << "dre " << dre->getNameInfo().getAsString() << "\n";
    			}
    		}
    		for (Stmt::child_iterator s_ci = s->child_begin(), s_ce = s->child_end(); s_ci != s_ce; ++s_ci) {
    			if(*s_ci){
    				replicate(*s_ci);
    			}

    		}
    }

    std::string getStmtText(Stmt *s) {
        SourceLocation a(SM->getExpansionLoc(s->getLocStart())), b(Lexer::getLocForEndOfToken(SourceLocation(SM->getExpansionLoc(s->getLocEnd())), 0,  *SM, *LO));
        return std::string(SM->getCharacterData(a), SM->getCharacterData(b)-SM->getCharacterData(a));
    }

};

class Replication : public ASTFrontendAction{
public:
	Replication(){}
	void EndSourceFileAction() override {
		TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs()); //write in a temporary (file, stream?)

	}

	std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
		TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
		return llvm::make_unique<MyReplicator>(&CI, &TheRewriter);
	}
private:
	Rewriter TheRewriter;
};


int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());
  return Tool.run(newFrontendActionFactory<Replication>().get());
}
