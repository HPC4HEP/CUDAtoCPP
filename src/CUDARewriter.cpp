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

////This class takes care of the universal replication, and of the movement of the declarations in the right places //fixme don't add thread_loops here? just the first?
//class MyReplicator : public ASTConsumer {
//public:
//	MyReplicator(CompilerInstance *comp, Rewriter * R) : ASTConsumer(), CI(comp), Rew(R) { }
//	virtual ~MyReplicator() { }
//	virtual void Initialize(ASTContext &Context) {
//		SM = &Context.getSourceManager();
//		LO = &CI->getLangOpts();
//		PP = &CI->getPreprocessor();
//	}
//
//	virtual bool HandleTopLevelDecl(DeclGroupRef DG) {
//
//
//		Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
//		//TODO what's the difference between SpellingLoc and Loc?
//		//SourceLocation loc = firstDecl->getLocation();
//		SourceLocation sloc = SM->getSpellingLoc(firstDecl->getLocation());
//
//		/*  Checking which file we are scanning.
//		 *  We skip everything apart from the main file.
//		 *  TODO: Rewriting some includes?
//		 *  FIXME: "extern" keyword in #defines could cause problems
//		 */
//		if( SM->getFileID(sloc) != SM->getMainFileID()){ //FileID mismatch
//			return true; //Just skip the loc but continue parsing. TODO: Can I skip the whole file?
//		}
//
//		//DEBUG std::cout << "HandleTopLevelDecl: " << sloc.printToString(*SM) << "\n";
//
//        //Walk declarations in group and rewrite
//        for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
//            //Handles globally defined functions
//            if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*i)) {
//            	//DEBUG std::cout << "\tFunctionDecl2 fd\n";
//            	//TODO: Don't translate explicit template specializations
//            	//fixme Ignoring templates for now
//            	//if(fd->getTemplatedKind() == clang::FunctionDecl::TK_NonTemplate || fd->getTemplatedKind() == FunctionDecl::TK_FunctionTemplate) {
//                    if (fd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>()) { //FIXME need device?
//                    	//DEBUG std::cout << "\t\tfd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>() = true\n";
//                    	//Device function, so rewrite kernel
//                        RewriteKernelFunction(fd);
//                    }
//                  //Templates again.
////                } else {
////                    if (fd->getTemplateSpecializationInfo())
////                    	std::cout << "DEBUG: fd->getTemplateSpecializationInfo = true (Skip?)\n";
////                    else
////                    	std::cout << "DEBUG: Non-rewriteable function without TemplateSpecializationInfo detected?\n";
////                }
//            }
//            //TODO rewrite type declarations
//        }
//        return true;
//}
//
//private:
//    CompilerInstance *CI;
//    SourceManager *SM;
//    LangOptions *LO;
//    Preprocessor *PP;
//    Rewriter *Rew;
//    std::set<std::string> KernelDecls;
//    std::vector<std::string> NewDecls;
//    SourceLocation kernelbodystart;
//
//    void RewriteKernelFunction(FunctionDecl* kf) {
//           //Parameter Rewriting for a kernel
//   //        if (kf->hasBody()) {
//           if (Stmt *body = kf->getBody()){
//           	//Rew->InsertTextBefore(SM->getExpansionLoc(body->getLocStart()),"prova1\n");
//           	//DEBUG std::cout << "\tkf->hasBody() = true\n";
//           	//TODO ReorderKernel(body);
//           	kernelbodystart = PP->getLocForEndOfToken(body->getLocStart());
//           	replicate(body);
//            std::string a = "\n";
//            for(int i = 0; i < NewDecls.size(); i++){
//               a += NewDecls[i] + "\n";
//            }
//               Rew->InsertTextAfter(kernelbodystart, a);
//           }
//           //std::cout << KernelVars.size() << "\n";
//
//
//    }
//    void replicate(Stmt *s){
//
//        	//if(CompoundStmt *cs = dyn_cast<CompoundStmt>(s)){
//        	//	for(Stmt::child_iterator i = cs->body_begin(), e = cs->body_end(); i!=e; ++i){
//        	//		if(*i){
//    		//declaration inside a compound statement, which is a new scope
//    		if (DeclStmt *ds = dyn_cast<DeclStmt>(s)){
//    			DeclGroupRef DG = ds->getDeclGroup();
//    			for (DeclGroupRef::iterator i2 = DG.begin(), e = DG.end(); i2!=e; ++i2){
//    			//for(clang::DeclGroupIterator i2 = ds->decl_begin(), e2 = ds->decl_end(); i2 != e2; ++i2){
//    				if(*i2){
//    					if(VarDecl *vd = dyn_cast<VarDecl>(*i2)){
//    						if(vd->hasAttr<CUDASharedAttr>()){
//    						//if (CUDASharedAttr *sharedAttr = vd->getAttr<CUDASharedAttr>()) {
////    							NewDecls.push_back(vd->getType().getAsString()+" "+vd->getNameAsString()+";");
////        						if(vd->hasInit()){
////        							std::cout << "\n" << vd->getNameAsString() << " has init!\n";
////        							Rew->ReplaceText(SourceRange(vd->getLocStart(), PP->getLocForEndOfToken(vd->getLocEnd())), vd->getNameAsString() + " = " + getStmtText(vd->getInit()) + ";");
////        						} else {
////        							Rew->ReplaceText(SourceRange(vd->getLocStart(), PP->getLocForEndOfToken(vd->getLocEnd())), "");
////        						}
//    							continue;
//    						} else {
//    						KernelDecls.insert(vd->getNameAsString());
//    						//std::cout << "inserted vd " << vd->getNameAsString() << "\n";
//    						NewDecls.push_back(vd->getType().getAsString()+" "+vd->getNameAsString()+"[numThreads];");
//    						if(vd->hasInit()){
//    							Rew->ReplaceText(SourceRange(vd->getLocStart(), PP->getLocForEndOfToken(vd->getLocEnd())), vd->getNameAsString() + "[tid] = " + getStmtText(vd->getInit()) + ";");
//    						} else {
//    							Rew->ReplaceText(SourceRange(vd->getLocStart(), PP->getLocForEndOfToken(vd->getLocEnd())), "");
//    						}
//    						}
//
//    					}
//    				}
//    			}
//    		}
//    		else if(DeclRefExpr *dre = dyn_cast<DeclRefExpr>(s)){
//    			//if(dre->getNameInfo().getAsString() isin decls) then "vectorize"
//
//    			//std::cout << "dre " << dre->getNameInfo().getAsString() << "\n";
//    			if(KernelDecls.find(dre->getNameInfo().getAsString()) != KernelDecls.end()){
//    				//Rew->InsertTextAfter(dre->getLocEnd(), "[tid]");
//    				Rew->ReplaceText(SourceRange(dre->getLocStart(), dre->getLocEnd()), dre->getNameInfo().getAsString()+"[tid]");
//    				//std::cout << "dre " << dre->getNameInfo().getAsString() << "\n";
//    			}
//    		}
//    		for (Stmt::child_iterator s_ci = s->child_begin(), s_ce = s->child_end(); s_ci != s_ce; ++s_ci) {
//    			if(*s_ci){
//    				replicate(*s_ci);
//    			}
//
//    		}
//    }
//
//    std::string getStmtText(Stmt *s) {
//        SourceLocation a(SM->getExpansionLoc(s->getLocStart())), b(Lexer::getLocForEndOfToken(SourceLocation(SM->getExpansionLoc(s->getLocEnd())), 0,  *SM, *LO));
//        return std::string(SM->getCharacterData(a), SM->getCharacterData(b)-SM->getCharacterData(a));
//    }
//
//};

class MyASTConsumer : public ASTConsumer {
public:
	MyASTConsumer(CompilerInstance *comp, Rewriter *R) : ASTConsumer(), CI(comp), Rew(R){ }
	virtual ~MyASTConsumer() { }

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
            if (DeclContext *dc = dyn_cast<DeclContext>(*i)) {
            	//DEBUG std::cout << "\tDeclContext dc\n";
                //Basically only handles C++ member functions
                for (DeclContext::decl_iterator di = dc->decls_begin(), de = dc->decls_end(); di != de; ++di) {
                    if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*di)) {
                    	//DEBUG std::cout << "\t\tFunctionDecl1 fd\n";
                    	//Prevent implicitly defined functions from being rewritten (since there's no source to rewrite..)
                        if (!fd->isImplicit()) {
                        	//DEBUG std::cout << "\t\t\tfd->isImplicit = false\n";
                        	//TODO: What case applies here? Simply removing those kind of functions?
                            RewriteHostFunction(fd);
                        } else {
                        	//DEBUG std::cout << "\t\t\tfd->isImplicit = true (skip)\n";
                        }
                    }
                }
            }
            //Handles globally defined functions
            if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*i)) {
            	//DEBUG std::cout << "\tFunctionDecl2 fd\n";
            	//TODO: Don't translate explicit template specializations
            	//fixme Ignoring templates for now
            	//if(fd->getTemplatedKind() == clang::FunctionDecl::TK_NonTemplate || fd->getTemplatedKind() == FunctionDecl::TK_FunctionTemplate) {
                    if (fd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>()) {
                    	//DEBUG std::cout << "\t\tfd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>() = true\n";
                    	//Device function, so rewrite kernel
                        RewriteKernelFunction(fd);
                        if (fd->hasAttr<CUDAHostAttr>()){
                        	//DEBUG std::cout << "\t\t\tfd->hasAttr<CUDAHostAttr>() = true\n";
                            //Also a host function, so rewrite host?
                            RewriteHostFunction(fd);
                        } else {
                        	//DEBUG std::cout << "\t\t\tfd->hasAttr<CUDAHostAttr>() = false\n";
                            //Simply a device function, so remove from host?
                            //RemoveFunction(fd, HostRewrite);
                        }
                    } else {
                    	//DEBUG std::cout << "\t\tfd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>() = false\n";
                        //Simply a host function, so rewrite...
                        RewriteHostFunction(fd);
//                        if (CUDAHostAttr *attr = fd->getAttr<CUDAHostAttr>()) {
//                        	std::cout << "DEBUG: Found CUDAHostAttr...";
//                            SourceLocation instLoc = SM->getExpansionLoc(attr->getLocation());
//                            std::cout << " at loc " << instLoc.printToString(*SM);
//                            SourceRange realRange(instLoc, PP->getLocForEndOfToken(instLoc));
//                            std::cout << " ending at " << PP->getLocForEndOfToken(instLoc).printToString(*SM);
//                            Rew->ReplaceText(instLoc, Rew->getRangeSize(realRange), "");
//                            std::cout << " Removed!\n";
//                        }
                        //...and remove from kernel?
//                        RemoveFunction(fd, KernelRewrite);

//                        if (fd->getNameAsString() == MainFuncName) {
//                            RewriteMain(fd);
//                        }
                    }
                  //Templates again.
//                } else {
//                    if (fd->getTemplateSpecializationInfo())
//                    	std::cout << "DEBUG: fd->getTemplateSpecializationInfo = true (Skip?)\n";
//                    else
//                    	std::cout << "DEBUG: Non-rewriteable function without TemplateSpecializationInfo detected?\n";
//                }
            //Globally defined variables
            } else if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
            	//DEBUG std::cout << "\tVarDecl vd\n";
                //RemoveVar(vd, KernelRewrite);
                RewriteHostVarDecl(vd);
            //Rewrite Structs here
            //Ideally, we should keep the expression inside parentheses ie __align__(<keep this>)
            // and just wrap it with __attribute__((aligned (<kept Expr>)))
            //TODO: Finish struct attribute translation
        	} else if (RecordDecl * rd = dyn_cast<RecordDecl>(*i)) {
        		//DEBUG std::cout << "\tRecordDecl rd\n";
                if (rd->hasAttrs()) {
                	//DEBUG std::cout << "\t\trd->hasAttrs = true\n";
//                    for (Decl::attr_iterator at = rd->attr_begin(), at_e = rd->attr_end(); at != at_e; ++at) {
//                        if (AlignedAttr *align = dyn_cast<AlignedAttr>(*at)) {
//                            if (!align->isAlignmentDependent()) {
//                                llvm::errs() << "Found an aligned struct of size: " << align->getAlignment(rd->getASTContext()) << " (bits)\n";
//                            } else {
//                                llvm::errs() << "Found a dependent alignment expresssion\n";
//                            }
//                        } else {
//                            llvm::errs() << "Found other attrib\n";
//                        }
//                    }
                }
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
    //std::set<VarDecl *> KernelDecls;
    std::set<std::string> initStmts;
    std::set<std::string> KernelDecls;
    std::vector<std::string> NewDecls;

    std::string TL_START1 = "for(threadIdx.z=0; threadIdx.z < blockDim.z; threadIdx.z++){\n";
    std::string TL_START2 = "for(threadIdx.y=0; threadIdx.y < blockDim.y; threadIdx.y++){\n";
    std::string TL_START3 = "for(threadIdx.x=0; threadIdx.x < blockDim.x; threadIdx.x++){\n";
    std::string TL_START = TL_START1+TL_START2+TL_START3+"tid=threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y;";
    //todo macro for tid instead of recalculating it everytime? it's the same maybe


    std::string TL_END = "}}}";

    /*
     * Simple function to strip attributes from host functions that may be declared as
     * both __host__ and __device__, then passes off to the host-side statement rewriter
     */
    void RewriteHostFunction(FunctionDecl *hostFunc) {
		SourceLocation sloc = SM->getSpellingLoc(hostFunc->getLocation());
		//DEBUG std::cout << "RewriteHostFunction hostFunc: " << sloc.printToString(*SM) << "\n";
    	//Remove any CUDA function attributes
        if (CUDAHostAttr *attr = hostFunc->getAttr<CUDAHostAttr>()) {
        	//DEBUG std::cout << "\thostFunc->hasAttr<CUDAHostAttr>() = true\n";
            RewriteAttr(attr, "", *Rew);
        }
        if (CUDADeviceAttr *attr = hostFunc->getAttr<CUDADeviceAttr>()) {
        	//DEBUG std::cout << "\thostFunc->hasAttr<CUDADeviceAttr>() = true\n";
            RewriteAttr(attr, "", *Rew);
        }

        //Rewrite the body
        if (Stmt *body = hostFunc->getBody()) {
        	//DEBUG std::cout << "\thostFunc->hasBody() = true\n";
            RewriteHostStmt(body);
        }
    }

    //Dispatching between expressions, declarations and other statements
    void RewriteHostStmt(Stmt *s) {
    	//DEBUG std::cout << "RewriteHostStmt s: ";
        if (Expr *e = dyn_cast<Expr>(s)) {
        	//DEBUG SourceLocation sloc = SM->getSpellingLoc(e->getExprLoc());
        	//DEBUG std::cout << "<Expr> " << sloc.printToString(*SM) << "\n";
        	std::string str;
        	if (RewriteHostExpr(e, str)) ReplaceStmtWithText(e, str, *Rew);
        }
        else if (DeclStmt *ds = dyn_cast<DeclStmt>(s)) {
        	//DEBUG SourceLocation sloc = SM->getSpellingLoc(ds->getStartLoc());
        	//DEBUG std::cout << "<DeclStmt> " << sloc.printToString(*SM) << "\n";
        	DeclGroupRef DG = ds->getDeclGroup();
			Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
            for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
                if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                	//DEBUG std::cout << "\tVarDecl vd\n";
                    RewriteHostVarDecl(vd);
                }
            }
        } else {
            //Traverse children and recurse
            for (Stmt::child_iterator CI = s->child_begin(), CE = s->child_end(); CI != CE; ++CI) {
            	//DEBUG std::cout << "children!\n";
                if (*CI) RewriteHostStmt(*CI);
            }
        }
    }

    void RewriteKernelFunction(FunctionDecl* kf) {
    	//DEBUG: SourceLocation sloc = SM->getSpellingLoc(kf->getLocation());
    	//DEBUG: std::cout << "RewriteKernelFunction kf: " << sloc.printToString(*SM) << "\n";

        if (CUDAGlobalAttr *attr = kf->getAttr<CUDAGlobalAttr>()) {
        	//DEBUG std::cout << "\tkf->hasAttr<CUDAGlobalAttr>() = true\n";
    		RewriteAttr(attr, "", *Rew);
        }
        if (CUDADeviceAttr *attr = kf->getAttr<CUDADeviceAttr>()) {
        	//DEBUG std::cout << "\tkf->hasAttr<CUDADeviceAttr>() = true\n";
            RewriteAttr(attr, "", *Rew);
        }

        if (CUDAHostAttr *attr = kf->getAttr<CUDAHostAttr>()) {
        	//DEBUG std::cout << "\tkf->hasAttr<CUDAHostAttr>() = true\n";
            RewriteAttr(attr, "", *Rew);
        }

        //Parameter Rewriting for a kernel
    	if (kf->hasAttr<CUDAGlobalAttr>()) {
    		//DEBUG std::cout << "\tkf->hasAttr<CUDAGlobalAttr>() = true\n";
        	//Means host callable
    		std::string SStr;
			llvm::raw_string_ostream S(SStr);
			S << kf->getCallResultType().getAsString() << " " << kf->getNameAsString() << "(";
			for( int j = 0; j < kf->getNumParams(); j++){
				//TODO Check if this is a general rule
				S << kf->getParamDecl(j)->getType().getAsString() << " " << kf->getParamDecl(0)->getQualifiedNameAsString() << ", ";
			}
			S << "dim3 gridDim, dim3 blockDim, uint3 blockIdx)";
			//DEBUG std::cout << S.str() << "\n";
			SourceLocation start = SM->getExpansionLoc(kf->getLocStart());
			SourceLocation end = PP->getLocForEndOfToken(SM->getExpansionLoc(kf->getParamDecl(kf->getNumParams()-1)->getLocEnd()));
			SourceRange range(start, end);//PP->getLocForEndOfToken(instLoc));
			Rew->ReplaceText(start, Rew->getRangeSize(range), S.str());


        }
//        if (kf->hasBody()) {
        if (Stmt *body = kf->getBody()){
        	//Rew->InsertTextBefore(SM->getExpansionLoc(body->getLocStart()),"prova1\n");
        	//DEBUG std::cout << "\tkf->hasBody() = true\n";
        	//TODO ReorderKernel(body);
        	//kernelbodystart = PP->getLocForEndOfToken(body->getLocStart());
        	//newbody = replicate(body);
            T3(body, true);
//            std::string a = "\n";
//            for(int i = 0; i < NewDecls.size(); i++){
//            	a += NewDecls[i] + "\n";
//            }
//            Rew->InsertTextAfter(kernelbodystart, a+"TL_START");
//            Rew->InsertTextBefore(body->getLocEnd(), "TL_END");
        }
        //std::cout << KernelVars.size() << "\n";


    }

//    void RewriteKernelStmt(Stmt *ks, int depth, bool first) {
//    	if(first){
//			Rew->InsertTextAfter(PP->getLocForEndOfToken(SM->getExpansionLoc(ks->getLocStart())), "\nTL_BEGIN\n");
//			Rew->InsertTextBefore(SM->getExpansionLoc(ks->getLocEnd()), "\nTL_END\n");
//    	}
//    	/*
//    	 * That's our main business!
//    	 * TODO: probably we will need some disambiguation of the variable names
//    	 * TODO: we have to split all the declarations with assignments in two different statements (declaration + assignment)
//    	 * TODO: we have to move all the declarations to the top of the kernel body
//    	 * TODO: we have to add the uint3/dim3 threadIdx declaration
//    	 * TODO: after the declarations, we have to surround everythin left of the body with the triple for loop (threadIdx.x,y,z)
//    	 * TODO: for every __syncthread() we encounter, we split the triple for-loop (also for break, continue, return statements?)
//    	 */
//        //Visit this node
//    	//std::cout << "Inserting stuff\n";
////    	if(depth == 1) {
////    	Rew->InsertTextBefore(SM->getExpansionLoc(ks->getLocStart()), "\n/* here they are the for loops */\n");
////    	}
//		if (Expr *e = dyn_cast<Expr>(ks)) {
//			std::cout << "Expr " << getStmtText(e) << " @ " << depth << "\n";
//			//SourceLocation sloc = SM->getSpellingLoc(e->getExprLoc());
//			//std::cout << "<Expr> " << sloc.printToString(*SM) << " " << e-> << "\n";
//			std::string str;
//			//FIXME it has to be shaped like this? Why RewriteKernelExpr() cannot just directly rewrite?
//			if (RewriteKernelExpr(e, str)) {
//				//Found a __syncthreads()
//				//Rew->InsertTextAfter(); //3-loop after the beginning of this compound stmt
//				ReplaceStmtWithText(e, str, *Rew); //simply comments the call?
//				//Rew->InsertTextBefore(); //closing before the __sync loc
//				//Rew->InsertTextAfter(); //3-loop after the __sync loc
//				//Rew->InsertTextBefore(); //closing before the end of this compound stmt
//			}
//		}
//		else if (DeclStmt *ds = dyn_cast<DeclStmt>(ks)) {
//			std::cout << "DeclStmt " << getStmtText(ds) << " @ " << depth << "\n";
//			//SourceLocation sloc = SM->getSpellingLoc(ds->getStartLoc());
//			//std::cout << "<DeclStmt> " << sloc.printToString(*SM) << " " << num << "\n";
//			DeclGroupRef DG = ds->getDeclGroup();
//			for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(), counter=0; i != e; ++i, counter++) {
//				//std::cout << "group " << counter << "\n";
//				if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
//					//DEBUG std::cout << "\tVarDecl vd\n";
//					//if(vd->hasInit()) std::cout << "init\n";
//
//					if(vd->hasInit()) {
//					//std::cout << vd->getQualifiedNameAsString() << "\n";
//					KernelDecls.insert(vd->getCanonicalDecl());
//					std::string stmtstr = vd->getNameAsString() + "=" + getStmtText(vd->getInit()) + ";\n";
//					initStmts.insert(stmtstr);
//					//std::cout << stmtstr << "\n";
//					} else {
//						//std::cout << vd->getType().getAsString() << " " << vd->getNameAsString() << "\n";
//						KernelDecls.insert(vd);
//					}
//
//					RewriteKernelVarDecl(vd);
//				}
//				//TODO other non-top level declarations??
//			}
//		}
//		//TODO rewrite any other Stmts?
////		else if(CompoundStmt *cs = dyn_cast<CompoundStmt>(ks)){
////				std::cout << "CompoundStmt "<< getStmtText(cs) << " @ " << depth << "\n";
////
////				//Entering a new scope
////				//useful to find new scopes! because is simply another level of depth...
////				//TODO: search for __syncthreads(), and if found, surround this statement with triple-for?
////				depth++;
////				//Traverse children and recurse
////				Rew->InsertTextAfter(PP->getLocForEndOfToken(cs->getLocStart()), "\nTL_BEGIN (CompoundStmt)\n");
////				for (Stmt::child_iterator CI = cs->body_begin(), CE = cs->body_end(); CI != CE; ++CI) {
////					//std::cout << "compound children:\n";
////					if (*CI) {
////						//std::cout << "RewriteKernelStmt level " << depth << ": "; // << getStmtText(*CI) << "\n";
////						RewriteKernelStmt(*CI, depth);
////					}
////				}
////				Rew->InsertTextBefore(cs->getRBracLoc(), "\nTL_END (CompoundStmt)\n");
////
////		}
////		else if(IfStmt *is = dyn_cast<IfStmt>(ks)){
////			std::cout << "IfStmt -> " << getStmtText(ks) << " @ " << depth << "\n";
////			for (Stmt::child_iterator CI = ks->child_begin(), CE = ks->child_end(); CI != CE; ++CI) {
////				//std::cout << "children\n";
////				if (*CI) {
////					RewriteKernelStmt(*CI, depth, false);
////				}
////			}
////		}
//		else {
////			if(ForStmt *fs = dyn_cast<ForStmt>(ks)){
////				if(fs->)
////				Rew->InsertTextBefore(fs->getLocStart(), "\nTL_END\n");
////				Rew->InsertTextAfter(fs->get)
////			}
//			//something different i.e. for loop
//			//Traverse children and recurse
//			//
//
//			//TODO THERE'S NO SUCH THING AS PURE COMPOUND STATEMENTS PROBABLY, SO WE NEED TO PUT A TL_END BEFORE THE STATEMENTS
//			// INVOLVING ANY OF THEM (i.e. before 'for(){}' ) AND A TL_BEGIN AFTER ANY OF THEM (after the '{' )
//			//TODO CHECK the advantages of doing the cast of compound inside the else branch in the for loop... could be useful
//			// to do it outside but still in the else, or add another specific case in the if-elseif chain for compound statements
//
//			//(at the moment the first TL_BEGIN at the beginning of the function body and the last TL_END at the end are missing.
//			//TODO: it can be done maybe simply checking the depth or there is a better way to automatize it (i.e. considering
//			// the body of the whole function as a compound statement?)
//			std::cout << "Stmt -> " << getStmtText(ks) << " @ " << depth << "\n";
//			for (Stmt::child_iterator CI = ks->child_begin(), CE = ks->child_end(); CI != CE; ++CI) {
//				//std::cout << "children\n";
//				if (*CI) {
//					if(CompoundStmt *cs = dyn_cast<CompoundStmt>(*CI)){
//						if(IfStmt *is = dyn_cast<IfStmt>(ks)){
//							//Compound statements of if statements doesn't have to be surrounded by thread_loops
//							RewriteKernelStmt(*CI, depth, false);
//						}
//						else{
//							std::cout << "CompoundStmt "<< getStmtText(cs) << " @ " << depth << "\n";
//							Rew->InsertTextBefore(SM->getExpansionLoc(ks->getLocStart()), "\nTL_END (Opening a CompoundStmt)\n");
//							//Entering a new scope
//							//useful to find new scopes! because is simply another level of depth...
//							//TODO: search for __syncthreads(), and if found, surround this statement with triple-for?
//							depth++;
//							//Traverse children and recurse
//							Rew->InsertTextAfter(PP->getLocForEndOfToken(cs->getLocStart()), "\nTL_BEGIN (CompoundStmt)\n");
//							for (Stmt::child_iterator CCI = cs->body_begin(), CCE = cs->body_end(); CCI != CCE; ++CCI) {
//								//std::cout << "compound children:\n";
//								if (*CCI) {
//									//if(BreakStmt *bs = dyn_cast<BreakStmt>(*CCI)) Rew->InsertTextAfter(ks->getLocEnd(), "Break goes here!\n");
//									//std::cout << "RewriteKernelStmt level " << depth << ": "; // << getStmtText(*CI) << "\n";
//									RewriteKernelStmt(*CCI, depth, false);
//								}
//							}
//							Rew->InsertTextBefore(cs->getRBracLoc(), "\nTL_END (CompoundStmt)\n");
//							//FIXME: bug with the semicolon ending a do-while statement
//							Rew->InsertTextAfter(PP->getLocForEndOfToken(SM->getExpansionLoc(ks->getLocEnd())), "\nTL_BEGIN (Closing a CompoundStmt)\n");
//						}
//					}
//					else {
//					//std::cout << "RewriteKernelStmt level " << depth << ": "; // << getStmtText(*CI) << "\n";
//						//if(BreakStmt *bs = dyn_cast<BreakStmt>(*CI))
//						//	Rew->InsertTextAfter(ks->getLocEnd(), "Break goes here!\n");
//						RewriteKernelStmt(*CI, depth, false);
//					}
//				}
//			}
//			//Rew->InsertTextAfter(PP->getLocForEndOfToken(SM->getExpansionLoc(ks->getLocEnd())), "\nTL_BEGIN (Stmt)\n");
//		}
//    }

    SourceLocation kernelbodystart;

    void T3(Stmt *s, bool first){
    	if(first){
    		SourceLocation begin = s->getLocStart();
    		if(CompoundStmt * cs = dyn_cast<CompoundStmt>(s)){
				for(Stmt::child_iterator i = cs->body_begin(), e = cs->body_end(); i!=e; ++i){
					if(*i){
						//std::cout << getStmtText(*i);
						//if (DeclStmt *ds = dyn_cast<DeclStmt>(*i)){
							//DeclGroupRef DG = ds->getDeclGroup();
							//for (DeclGroupRef::iterator i2 = DG.begin(), e = DG.end(); i2!=e; ++i2){
								//if(*i2){
									if(DeclStmt *vd = dyn_cast<DeclStmt>(*i)){
										begin = PP->getLocForEndOfToken(vd->getLocEnd());
										//std::cout << "ueue\n";
									}
									else break;
								//}
							//}
						//} else {
						//	break;
						//}
					}
				}
    		} else {
    			std::cout << "\nil primo non era un compound!\n";
    			first = false;

    		}
    		Rew->InsertTextAfter(begin, "\n"+TL_START+"\n");
    		Rew->InsertTextBefore(s->getLocEnd(), "\n"+TL_END+"\n");
    		first = false;
    	}
    	//if(CompoundStmt *cs = dyn_cast<CompoundStmt>(s)){
    	//	for(Stmt::child_iterator i = cs->body_begin(), e = cs->body_end(); i!=e; ++i){
    	//		if(*i){
		//declaration inside a compound statement, which is a new scope
//		if (DeclStmt *ds = dyn_cast<DeclStmt>(s)){
//			DeclGroupRef DG = ds->getDeclGroup();
//			for (DeclGroupRef::iterator i2 = DG.begin(), e = DG.end(); i2!=e; ++i2){
//			//for(clang::DeclGroupIterator i2 = ds->decl_begin(), e2 = ds->decl_end(); i2 != e2; ++i2){
//				if(*i2){
//					if(VarDecl *vd = dyn_cast<VarDecl>(*i2)){
//						KernelDecls.insert(vd->getNameAsString());
//						//std::cout << "inserted vd " << vd->getNameAsString() << "\n";
//						NewDecls.push_back(vd->getType().getAsString()+" "+vd->getNameAsString()+"[numThreads];");
//						if(vd->hasInit()){
//							Rew->ReplaceText(SourceRange(vd->getLocStart(), PP->getLocForEndOfToken(vd->getLocEnd())), vd->getNameAsString() + "[tid] = " + getStmtText(vd->getInit()) + ";");
//						} else {
//							Rew->ReplaceText(SourceRange(vd->getLocStart(), PP->getLocForEndOfToken(vd->getLocEnd())), "");
//						}
//
//					}
//				}
//			}
//		}
//		else if(DeclRefExpr *dre = dyn_cast<DeclRefExpr>(s)){
//			//if(dre->getNameInfo().getAsString() isin decls) then "vectorize"
//
//			//std::cout << "dre " << dre->getNameInfo().getAsString() << "\n";
//			if(KernelDecls.find(dre->getNameInfo().getAsString()) != KernelDecls.end()){
//				//Rew->InsertTextAfter(dre->getLocEnd(), "[tid]");
//				Rew->ReplaceText(SourceRange(dre->getLocStart(), dre->getLocEnd()), dre->getNameInfo().getAsString()+"[tid]");
//				//std::cout << "dre " << dre->getNameInfo().getAsString() << "\n";
//			}
//		}
		//}
    	//	}
    	//}
//    	if (DeclStmt *ds = dyn_cast<DeclStmt>(s)) {
//			SourceLocation sloc = SM->getSpellingLoc(ds->getStartLoc());
//			std::cout << "<DeclStmt> " << sloc.printToString(*SM) << " \n";
//			DeclGroupRef DG = ds->getDeclGroup();
//			for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(), counter=0; i != e; ++i, counter++) {
//				std::cout << "group " << counter << "\n";
//				if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
//					std::cout << "\tVarDecl vd\n";
//
//					if(vd->hasInit()) {
//					std::cout << vd->getType().getAsString() << " " << vd->getQualifiedNameAsString() << "[];\n";
//					KernelDecls.insert(vd->getCanonicalDecl());
//					std::string stmtstr = "TL_START\n" + vd->getNameAsString() + "[] = " + getStmtText(vd->getInit()) + ";\nTL_END\n";
//					initStmts.insert(stmtstr);
//					std::cout << stmtstr << "\n";
//					} else {
//						std::cout << vd->getType().getAsString() << " " << vd->getNameAsString() << "[];\n";
//						KernelDecls.insert(vd);
//					}
//
//					RewriteKernelVarDecl(vd);
//				}
//				//TODO other non-top level declarations??
//			}
//		}
    	//This big if searches for compound statements that can contain a syncthread call
    	if(IfStmt * is = dyn_cast<IfStmt>(s)){ //Candidate: IF statement
    		bool sync_then = false, sync_else = false; //We need this booleans after to start the rewriting if we found some syncthreads inside

    		SourceLocation loc_start_if = is->getLocStart();
    		SourceLocation loc_end_if = is->getLocEnd();

    		SourceLocation loc_start_then;
			SourceLocation loc_end_then;
			SourceLocation loc_start_else;
			SourceLocation loc_end_else;

			Expr * cond;
			std::set<std::pair<SourceLocation, SourceLocation>> loc_then_sync; //With this two structures we memorize pairs containing the location before and after the syncthreads
			std::set<std::pair<SourceLocation, SourceLocation>> loc_else_sync;

			std::string newif = "";

			if(Stmt * sts = is->getThen()){ //Then branch is not null
				if(CompoundStmt* ts = dyn_cast<CompoundStmt>(sts)){ //Then branch is a compound statement
					loc_start_then = PP->getLocForEndOfToken(ts->getLocStart()); //We set the location after the bracket as then branch starting location
					loc_end_then = ts->getLocEnd(); //And we keep track of where the then branch ends
					for(Stmt::child_iterator ts_ci = ts->body_begin(), ts_ce = ts->body_end(); ts_ci != ts_ce; ++ts_ci){ //We iterate among the instructions inside the then branch
						if(*ts_ci){ //if this child is not null
							if(CallExpr *tce = dyn_cast<CallExpr>(*ts_ci)){//we search for function calls
								if(tce->getDirectCallee()->getNameAsString() == "__syncthreads"){ //and in particular if the call is a syncthreads(), we keep track of it
									sync_then = true;
									loc_then_sync.insert(std::make_pair(tce->getLocStart(), PP->getLocForEndOfToken(PP->getLocForEndOfToken(tce->getLocEnd())))); //we cross the double token at the end: ')' and ';'
								}
							}
						}
					}
				} else { // special case : the cast to compound statement failded, it means that this then branch is a single instruction (without {})
					if(CallExpr *tce = dyn_cast<CallExpr>(sts)){ //again we check if it's a call
						if(tce->getDirectCallee()->getNameAsString() == "__syncthreads"){ //and if it's a syncthreads
							sync_then = true;
							loc_start_then = PP->getLocForEndOfToken(PP->getLocForEndOfToken(sts->getLocStart())); //we just use the whole then branch locations (double token crossing again)
							loc_end_then = sts->getLocEnd();
						} else { // it was a function call but not a syncthreads
							loc_start_then = sts->getLocStart(); //i have to keep the content
							if(Stmt * ses = is->getElse()) loc_end_then = is->getElseLoc(); else loc_end_then = is->getLocEnd(); //if there's an else branch also, we use its loc beginning as a loc ending for the then, otherwise the end is the end of the whole if stmt
						}
					} else { //neither a function call and of course not a syncthreads
						loc_start_then = sts->getLocStart(); //keeping the content also in this case
						if(Stmt * ses = is->getElse()) loc_end_then = is->getElseLoc(); else loc_end_then = is->getLocEnd(); // again using else branch if it exists
					}
				}
				if(Stmt * ses = is->getElse()){ //Also the else branch is not null
					if(CompoundStmt * es = dyn_cast<CompoundStmt>(ses)){ //Else branch is a compound statement
						loc_start_else = PP->getLocForEndOfToken(es->getLocStart()); //We set the location after the bracket as the else branch starting location
						loc_end_else = es->getLocEnd(); //and we keep track of where the else branch ends
						for(Stmt::child_iterator es_ci = es->body_begin(), es_ce = es->body_end(); es_ci != es_ce; ++es_ci){ //We iterate among the instructions inside of the else branch
							if(*es_ci){ //if this child is not null
								if(CallExpr *ece = dyn_cast<CallExpr>(*es_ci)){ //we search for function calls
									if(ece->getDirectCallee()->getNameAsString() == "__syncthreads"){ //and in particular if the call is a syncthreads(), we keep track of it
										sync_else = true;
										loc_else_sync.insert(std::make_pair(ece->getLocStart(), PP->getLocForEndOfToken(PP->getLocForEndOfToken(ece->getLocEnd())))); //we cross the double token at the end: ')' and ';'
									}
								}
							}
						}
					} else { //special case: the cast to compound statement failed, it means that this else branch is a single instruction (without {})
						if(CallExpr *tce = dyn_cast<CallExpr>(ses)){ //again we check if it's a call
							if(tce->getDirectCallee()->getNameAsString() == "__syncthreads"){ //and if it's a syncthreads
								sync_else = true;
								loc_start_else = PP->getLocForEndOfToken(PP->getLocForEndOfToken(ses->getLocStart())); //we just use the whole else branch locations (double token crossing)
								loc_end_else = ses->getLocEnd();
							} else { //it was a function call but not a synchtreads
								loc_start_else = ses->getLocStart(); //we want to keep the content
								loc_end_else = PP->getLocForEndOfToken(PP->getLocForEndOfToken(SM->getExpansionLoc(ses->getLocEnd()))); //the end of the else branch
							}
						} else { //neither a function call and of course not a synchthreads
							loc_start_else = ses->getLocStart(); //keeping the conent also in this case
							loc_end_else = PP->getLocForEndOfToken(PP->getLocForEndOfToken(SM->getExpansionLoc(ses->getLocEnd()))); //the end of the else branch
						}
					}
				}
			} // else { } //is it possibile a null then branch even if the IF statement wasn't null?


			if(sync_then || sync_else){ //We found some syncthreads
				cond = is->getCond(); //we keep track of the if condition
				loc_then_sync.insert(std::make_pair(loc_end_then, loc_end_then)); //we add the ending locations of the then and else branches in the structures (useful for the last syncthreads found)
				loc_else_sync.insert(std::make_pair(loc_end_else, loc_end_else));
			}
			//Rewritings
			if(sync_then){ //syncthreads in the then branch
				//Before the starting location, we add the declaration of a boolean that will keep the value of the condition, this is needed to avoid side effects when
				//there are statements inside that modify the value of the condition
				//We also initialize this new boolean in a thread loop (we assume that the condition can be thread dependent)
				newif += "\n"+TL_END+"\nbool oldcond[numThreads];\n"+TL_START+"\noldcond[tid]=("+getStmtText(cond)+");\n"+TL_END+"\n";
				//OLD//Rew->InsertTextBefore(loc_start_if, "\nTL_END\nbool oldcond;\nTL_START\noldcond="+getStmtText(cond)+";\nTL_END\n");
				if(sync_else){ //syncthreads also in the else branch
					//We iterate among the location pairs of the synchtreads we found in both branches.
					for (std::set<std::pair<SourceLocation,SourceLocation>>::iterator i = loc_then_sync.begin(), e = loc_then_sync.end(), i2 = loc_else_sync.begin(), e2 = loc_else_sync.end();
							i != e || i2 != e2;
							/*noincrementhere*/) {
						if(i == loc_then_sync.begin()){ //The first location
							StringRef tcontent = Lexer::getSourceText(CharSourceRange(SourceRange(loc_start_then, (*i).first),false), *SM, *LO); //We take everything between the beginning and the location before the first syncthreads in the then branch
							StringRef econtent = Lexer::getSourceText(CharSourceRange(SourceRange(loc_start_else, (*i2).first),false), *SM, *LO); //We take everything between the beginning and the location before the first syncthreads in the else branch
							newif += TL_START+"\nif(oldcond[tid]){" + tcontent.str() + "\n"; //And we create a new if, with a new then branch
							//OLD//std::cout << "TL_START\nif(oldcond){" << tcontent.str() << "\n"; //And we create a new if, with a new then branch
							newif +="} else {\n" + econtent.str() + "}\n"+TL_END+"\n"; //and a new else branch, everything wrapped inside a thread_loop
							//OLD//std::cout << "} else {\n" << econtent.str() << "}\nTL_END\n"; //and a new else branch, everything wrapped inside a thread_loop
						} else { //The general scenario
								StringRef tcontent = Lexer::getSourceText(CharSourceRange(SourceRange((*(std::prev(i,1))).second, (*i).first),false), *SM, *LO); //We take everything between the end of a syncthread and the beginning of the next one
								StringRef econtent = Lexer::getSourceText(CharSourceRange(SourceRange((*(std::prev(i2,1))).second, (*i2).first),false), *SM, *LO); //same
								newif+= TL_START+"\nif(oldcond[tid]){" + tcontent.str() + "\n"; //And we create a new if, with a new then branch
								newif+= "} else {\n" + econtent.str()+ "}\n"+TL_END+"\n"; //and a new else branch, everything wrapped inside a thread_loop
								//OLD//std::cout <<  "TL_START\nif(oldcond){" << tcontent.str() << "\n"; //And we create a new if, with a new then branch
								//OLD//std::cout << "} else {\n" << econtent.str() << "}\nTL_END\n"; //and a new else branch, everything wrapped inside a thread_loop
						}
						if(i != e) ++i;//there is still something in the then branch
						if(i2!=e2) ++i2; //there is still something in the else branch
					}
				} else { //syncthreads only in the then branch
					for (std::set<std::pair<SourceLocation,SourceLocation>>::iterator i = loc_then_sync.begin(), e = loc_then_sync.end(); i != e; i++) { //We iterate among the location pairs
						if(i == loc_then_sync.begin()){ //The first location
							StringRef content = Lexer::getSourceText(CharSourceRange(SourceRange(loc_start_then, (*i).first),false), *SM, *LO); //We take everything between the beginning and the location before the first syncthreads in the then branch
							newif += ""+TL_START+"\nif(oldcond[tid]){" + content.str();
							//OLD//std::cout << "TL_START\nif(oldcond){" << content.str();
							if(is->getElse()){//it there is an else branch, even without syncthreads in it, we have to attach it
								StringRef allelse = Lexer::getSourceText(CharSourceRange(SourceRange(loc_start_else, loc_end_else), false), *SM, *LO);
								newif += "\n} else {\n" + allelse.str() + "}\n"+TL_END+"\n";
								//OLD//std::cout << "\n} else {\n" << allelse.str() << "}\nTL_END\n";
							} else { //there was no else branch at all, we just close
								newif+= "\n}"+TL_END+"\n";
								//OLD//std::cout << "\n}TL_END\n";
							}
					    } else { //General scenario, we keep adding pieces of the former then statement, creating new if stmts
							StringRef content = Lexer::getSourceText(CharSourceRange(SourceRange((*(std::prev(i,1))).second, (*i).first),false), *SM, *LO);
							newif += ""+TL_START+"\nif(oldcond[tid]){" + content.str() + "}\n"+TL_END+"\n";
							//OLD//std::cout << "TL_START\nif(oldcond){" << content.str() << "}\nTL_END\n";
						}
					}
				}
			} else if (sync_else){ //syncthreads only in the else branch
				//Before the starting location, we add the declaration of a boolean that will keep the value of the condition, this is needed to avoid side effects when
				//there are statements inside that modify the value of the condition
				//We also initialize this new boolean in a thread loop (we assume that the condition can be thread dependent)
				newif+="\n"+TL_END+"\nbool oldcond[numThreads];\n"+TL_START+"\noldcond[tid]=("+getStmtText(cond)+");\n"+TL_END+"\n";
				//OLD//Rew->InsertTextBefore(loc_start_if, "\nTL_END\nbool oldcond;\nTL_START\noldcond="+getStmtText(cond)+";\nTL_END\n");
				for(std::set<std::pair<SourceLocation, SourceLocation>>::iterator i = loc_else_sync.begin(), e = loc_else_sync.end(); i != e; i++){ //We iterate among the location pairs
					if(i == loc_else_sync.begin()){ //First Location
						StringRef allthen = Lexer::getSourceText(CharSourceRange(SourceRange(loc_start_then, loc_end_then), false), *SM, *LO); //We have to put the then branch before
						StringRef content = Lexer::getSourceText(CharSourceRange(SourceRange(loc_start_else, (*i).first),false), *SM, *LO);
						newif += "\n"+TL_START+"\nif(oldcond[tid]){"+ allthen.str() + "} else {\n" + content.str() + "}\n"+TL_END+"\n";
						//OLD//std::cout << "\nTL_START\nif(oldcond){"<< allthen.str() << "} else {\n" << content.str() << "}\nTL_END\n";
					} else { //General scenario
						StringRef content = Lexer::getSourceText(CharSourceRange(SourceRange((*(std::prev(i,1))).second, (*i).first),false), *SM, *LO);
						newif += ""+TL_START+"\nif(oldcond[tid]){} else{\n" + content.str() + "}\n"+TL_END+"\n"; //else stmts cannot live alone, so we add an empty then branch (TODO? FIXME? just using the negation of the oldcond? is it a correct translation though?)
						//OLD//std::cout << "TL_START\nif(oldcond){} else{\n" << content.str() << "}\nTL_END\n"; //else stmts cannot live alone, so we add an empty then branch (TODO? FIXME? just using the negation of the oldcond? is it a correct translation though?)
					}
				}
			} else { //there were no syncthreads in this if statement
				//TODO we have to iterate inside the if statement, to find possible statements that can contain syncthreads?
			}
			if(sync_then || sync_else){ //We found some syncthreads
				Rew->ReplaceText(SourceRange(is->getLocStart(), is->getLocEnd()), newif+"\n"+TL_START+"");
			}
    	} else if(WhileStmt * ws = dyn_cast<WhileStmt>(s)){
    		bool sync = false;

    		SourceLocation loc_start_while;
    		SourceLocation loc_end_while;
			Expr * cond;
			std::set<std::pair<SourceLocation, SourceLocation>> loc_sync;
			std::string newwhile = "";

    		if(Stmt * swb = ws->getBody()){
    			if(CompoundStmt * wb = dyn_cast<CompoundStmt>(swb)){
        			loc_start_while = wb->getLocStart();
        			loc_end_while = wb->getLocEnd();
    				for(Stmt::child_iterator wb_ci = wb->body_begin(), wb_ce = wb->body_end(); wb_ci != wb_ce; ++wb_ci){
    					if(*wb_ci){
    						if(CallExpr *wce = dyn_cast<CallExpr>(*wb_ci)){
    							if(wce->getDirectCallee()->getNameAsString() == "__syncthreads"){
    								//Found a synchtread inside the whilebody!
    								sync = true;
									loc_sync.insert(std::make_pair(wce->getLocStart(), PP->getLocForEndOfToken(PP->getLocForEndOfToken(wce->getLocEnd()))));
    							}
    						}
    					}
    				}
    			} else { //TODO While Statement with the body being only one instruction (not a block)
    				if(CallExpr *wce = dyn_cast<CallExpr>(swb)){
    				    if(wce->getDirectCallee()->getNameAsString() == "__syncthreads"){
    				    	//we encountered something like while(condition) __syncthreads();
    				    }
    				}
    			}
    		} else { //empty body, is that possible?

    		}
    		if(sync){
    			loc_sync.insert(std::make_pair(loc_end_while, loc_end_while));
    			cond = ws->getCond();
    			//Declaring and initializing the condition, inserting the label
    			newwhile += "\n"+TL_END+"\nbool cond1[numThreads];\n"+TL_START+"\ncond1[tid]=("+getStmtText(cond)+");\n"+TL_END+"\nLABEL: ";
    			//OLD//Rew->InsertTextBefore(ws->getLocStart(), "\nTL_END\nbool cond1;\nTL_START\ncond1=("+getStmtText(cond)+");\nTL_END\nLABEL:");
    			//Rew->ReplaceText(SourceRange StringRef)
				for(std::set<std::pair<SourceLocation, SourceLocation>>::iterator i = loc_sync.begin(), e = loc_sync.end(); i != e; i++){
					if(i == loc_sync.begin()){
						StringRef content = Lexer::getSourceText(CharSourceRange(SourceRange(loc_start_while, (*i).first),false), *SM, *LO);
						newwhile += ""+TL_START+"\nif(cond1[tid])" + content.str() + "}\n"+TL_END+"\n";
						//OLD//std::cout << "TL_START\nif(cond1){" << content.str() << "}\nTL_END\n";
					} else {
						StringRef content = Lexer::getSourceText(CharSourceRange(SourceRange((*(std::prev(i,1))).second, (*i).first),false), *SM, *LO);
						newwhile += ""+TL_START+"\nif(cond1[tid]){" + content.str() + "}\n"+TL_END+"\n";
						//OLD//std::cout << "TL_START\nif(cond1){" << content.str() << "}\nTL_END\n";
					}
				}
				//update condition, check if there are threads that can iterate again, and goto
    			newwhile += "\nbool go = false;\n"+TL_START+"\ncond1[tid]=("+getStmtText(cond)+");\nif(cond1[tid]) go=true;\n"+TL_END+"\nif(go) goto LABEL;\n";
				//OLD//std::cout << "\nbool go = false;\nTL_START\ncond1=("+getStmtText(cond)+");\nif(cond1) go=true;\nTL_END\nif(go) goto LABEL;\n";
				//std::cout << "\nif threads left goto label\n";
    			Rew->ReplaceText(SourceRange(ws->getLocStart(), ws->getLocEnd()), newwhile+"\n"+TL_START+"");
    		}
    	} else if (ForStmt * fs = dyn_cast<ForStmt>(s)){//For Statement
    		//todo should we take into account the side effects? if yes same procedure of the while loop; special case: how we handle the declaration of the index variable in the for?
    		bool sync = false;

			SourceLocation loc_start_for;
			SourceLocation loc_end_for;
			Expr * cond;
			Expr * inc;
			Stmt * init;
			std::set<std::pair<SourceLocation, SourceLocation>> loc_sync;
			std::string newfor = "";
    		if(Stmt * sfb = fs->getBody()){
    			if(CompoundStmt * fb = dyn_cast<CompoundStmt>(sfb)){
        			loc_start_for = fb->getLocStart();
        			loc_end_for = fb->getLocEnd();
    				for(Stmt::child_iterator fb_ci = fb->body_begin(), fb_ce = fb->body_end(); fb_ci != fb_ce; ++fb_ci){
    					if(*fb_ci){
    						if(CallExpr *fce = dyn_cast<CallExpr>(*fb_ci)){
    							if(fce->getDirectCallee()->getNameAsString() == "__syncthreads"){
    								//Found a synchtread inside the for body!
    								sync = true;
									loc_sync.insert(std::make_pair(fce->getLocStart(), PP->getLocForEndOfToken(PP->getLocForEndOfToken(fce->getLocEnd()))));
    							}
    						}
    					}
    				}
    			} else { //TODO For Statement with the body being only one instruction (not a block)
    				if(CallExpr *fce = dyn_cast<CallExpr>(sfb)){
    				    if(fce->getDirectCallee()->getNameAsString() == "__syncthreads"){
    				    	//for(condition) __syncthreads();
    				    }
    				}
    			}
    		} else { //empty body, is that possible?

    		}
    		if(sync){
    			loc_sync.insert(std::make_pair(loc_end_for, loc_end_for));
    			cond = fs->getCond();
    			inc = fs->getInc();
    			init = fs->getInit();
    			if (DeclStmt *vd = dyn_cast<DeclStmt>(init)) {
    				if(vd->isSingleDecl()){
    					std::cout << "for con dichiarazione nell'inizializzazione!!!\n";
    				} else {

    				}
    			}
    			newfor += "\n"+TL_END+"\nbool cond1[numThreads];\n"+TL_START+"\n"+getStmtText(init)+"\ncond1[tid]=("+getStmtText(cond)+");\n"+TL_END+"\nLABEL: ";
    			//OLD//Rew->InsertTextBefore(fs->getLocStart(), "\nTL_END\nbool cond1;\nTL_START\n"+getStmtText(init)+"\ncond1=("+getStmtText(cond)+");\nTL_END\nLABEL:");
				for(std::set<std::pair<SourceLocation, SourceLocation>>::iterator i = loc_sync.begin(), e = loc_sync.end(); i != e; i++){
					if(i == loc_sync.begin()){
						StringRef content = Lexer::getSourceText(CharSourceRange(SourceRange(loc_start_for, (*i).first),false), *SM, *LO);
						newfor += ""+TL_START+"\nif(cond1[tid])" + content.str() + "}\n"+TL_END+"\n";
						//OLD//std::cout << "TL_START\nif(cond1){" << content.str() << "}\nTL_END\n";
					} else {
						StringRef content = Lexer::getSourceText(CharSourceRange(SourceRange((*(std::prev(i,1))).second, (*i).first),false), *SM, *LO);
						newfor += ""+TL_START+"\nif(cond1[tid]){" + content.str() +"}\n"+TL_END+"\n";
						//OLD//std::cout << "TL_START\nif(cond1){" << content.str() <<"}\nTL_END\n";
					}
				}
				//update condition, check if there are threads that can iterate again, and goto
				newfor += "\nbool go = false;\n"+TL_START+"\n" + getStmtText(inc) + ";\ncond1[tid]=("+getStmtText(cond)+");\nif(cond1[tid]) go=true;\n"+TL_END+"\nif(go) goto LABEL;\n";
				//OLD//std::cout << "\nbool go = false;\nTL_START\n" << getStmtText(inc) << ";\ncond1=("+getStmtText(cond)+");\nif(cond1) go=true;\nTL_END\nif(go) goto LABEL;\n";
				//std::cout << "\nif threads left goto label\n";
				Rew->ReplaceText(SourceRange(fs->getLocStart(), fs->getLocEnd()), newfor+"\n"+TL_START+"");
    		}
//    		if(sync){
//				loc_sync.insert(std::make_pair(loc_end_for, loc_end_for));
//				cond = fs->getCond();
//				inc = fs->getInc();
//				init = fs->getInit();
//				//Declaring and initializing the condition, inserting the label
//				Rew->InsertTextBefore(fs->getLocStart(), "\nTL_END\nbool cond1;\nTL_START\ncond1=("+getStmtText(cond)+");\nTL_END\nLABEL:");
//				//Rew->ReplaceText(SourceRange StringRef)
//				for(std::set<std::pair<SourceLocation, SourceLocation>>::iterator i = loc_sync.begin(), e = loc_sync.end(); i != e; i++){
//					if(i == loc_sync.begin()){
//						StringRef content = Lexer::getSourceText(CharSourceRange(SourceRange(loc_start_for, (*i).first),false), *SM, *LO);
//						std::cout << "TL_START\nif(cond1){" << content.str() << "}\nTL_END\n";
//					} else {
//						StringRef content = Lexer::getSourceText(CharSourceRange(SourceRange((*(std::prev(i,1))).second, (*i).first),false), *SM, *LO);
//						std::cout << "TL_START\nif(cond1){" << content.str() << "}\nTL_END\n";
//					}
//				}
//				//update condition, check if there are threads that can iterate again, and goto
//				std::cout << "\nbool go = false;\nTL_START\ncond1=("+getStmtText(cond)+");\nif(cond1) go=true;\nTL_END\nif(go) goto LABEL;\n";
//				//std::cout << "\nif threads left goto label\n";
//			}

    	} else if (CallExpr *ce = dyn_cast<CallExpr>(s)){ //normal syncthreads simply in the body
    		if(ce->getDirectCallee()->getNameAsString() == "__syncthreads"){
    			//simply closing and reopening a thread loop
    			std::string a = "\n"+TL_END+"\n//"+getStmtText(ce)+";\n"+TL_START+"\n";
    			ReplaceStmtWithText(ce, a, *Rew);
    		}

    	} else { //other stmts
		for (Stmt::child_iterator s_ci = s->child_begin(), s_ce = s->child_end(); s_ci != s_ce; ++s_ci) {
			if(*s_ci){
				T3(*s_ci, false);
			}

		}

    	}
    }
//
//    bool RewriteKernelExpr(Expr *e, std::string &newExpr) {
//    	//DEBUG SourceLocation sloc = SM->getSpellingLoc(e->getExprLoc());
//    	//DEBUG std::cout << "RewriteKernelExpr e: " << sloc.printToString(*SM) << "\n";
//    	//TODO Loop Fission algorithm (Stage 2) should be triggered here
//    	//TODO but how to see everything in the scope of the __syncthreads() expr?
//    	//TODO also data buffering (stage 3) in variable usage (exprs) should be applied here.
//    	if (CallExpr *ce = dyn_cast<CallExpr>(e)){
//    		std::string funcName = ce->getDirectCallee()->getNameAsString();
//    		if (funcName == "__syncthreads") {
//    			//TODO remove the final ";"
//    			newExpr = "\nTL_END\n//"+getStmtText(ce)+";\nTL_BEGIN\n";
//    		}
//    	} else {
//    		return false;
//    	}
//    	return true;
//    }

    void RewriteKernelVarDecl(VarDecl *var) {
    	//DEBUG SourceLocation sloc = SM->getSpellingLoc(var->getLocation());
    	//DEBUG std::cout << "RewriteKernelVarDecl var: " << sloc.printToString(*SM) << "\n";
    	//TODO Handle shared memory variables/pointers
        if (CUDASharedAttr *sharedAttr = var->getAttr<CUDASharedAttr>()) {
        	//DEBUG std::cout << "\tvar->hasAttr<CUDASharedAttr>() = true\n";
            RewriteAttr(sharedAttr, "", *Rew);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>()){
            	//DEBUG std::cout << "\t\tvar->hasAttr<CUDADeviceAttr>() = true\n";
            	RewriteAttr(devAttr, "", *Rew);
            }
            //TODO rewrite extern shared mem
            //if (var->isExtern())?
        }
        //for(int i=0; i < KernelVars.c)
        //TODO Data buffering should be applied here (Stage 3)
        //TODO How to recognize which kind of variables we want to "vectorize"?
        //TODO How do we do it?

    }

    void RewriteHostVarDecl(VarDecl* var){
    	//DEBUG SourceLocation sloc = SM->getSpellingLoc(var->getLocation());
    	//DEBUG std::cout << "RewriteHostVarDecl var: " << sloc.printToString(*SM) << "\n";
    	if (CUDAConstantAttr *constAttr = var->getAttr<CUDAConstantAttr>()) {
    		//DEBUG std::cout << "\tvar->hasAttr<CUDAConstantAttr>() = true\n";
    		//TODO: Do something with __constant__ memory declarations
            RewriteAttr(constAttr, "", *Rew);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>()){
            	//DEBUG std::cout << "\t\tvar->hasAttr<CUDADeviceAttr>() = true\n";
            	RewriteAttr(devAttr, "", *Rew);
            }
//            //DeviceMemVars.insert(var);
//            ConstMemVars.insert(var);
//
//            TypeLoc origTL = var->getTypeSourceInfo()->getTypeLoc();
//            if (LastLoc.isNull() || origTL.getBeginLoc() != LastLoc.getBeginLoc()) {
//                LastLoc = origTL;
//                RewriteType(origTL, "cl_mem", HostRewrite);
//            }
//            return;
        }
        else if (CUDASharedAttr *sharedAttr = var->getAttr<CUDASharedAttr>()) {
        	//DEBUG std::cout << "\tvar->hasAttr<CUDASharedAttr>() = true\n";
            //Handle __shared__ memory declarations
            RewriteAttr(sharedAttr, "", *Rew);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>()){
            	//DEBUG std::cout << "\t\ttvar->hasAttr<CUDADeviceAttr>() = true\n";
            	RewriteAttr(devAttr, "", *Rew);
            }
//            //TODO rewrite shared mem
//            //If extern, remove extern keyword and make into pointer
//            //if (var->isExtern())
//            SharedMemVars.insert(var);
        }
        else if (CUDADeviceAttr *attr = var->getAttr<CUDADeviceAttr>()) {
        	//DEBUG std::cout << "\tvar->hasAttr<CUDADeviceAttr>() = true\n";
            //Handle __device__ memory declarations
            RewriteAttr(attr, "", *Rew);
            //TODO add to devmems, rewrite type
        }
    }

    bool RewriteHostExpr(Expr *e, std::string &newExpr) {
    	//SourceLocation sloc = SM->getSpellingLoc(e->getExprLoc());
    	//std::cout << "RewriteHostExpr e: " << sloc.printToString(*SM) << "\n";

        SourceRange realRange(SM->getExpansionLoc(e->getLocStart()),
                              SM->getExpansionLoc(e->getLocEnd()));

        //Rewriter used for rewriting subexpressions
        Rewriter exprRewriter(*SM, *LO);

        if (clang::CUDAKernelCallExpr *kce = dyn_cast<clang::CUDAKernelCallExpr>(e)) {
        	//DEBUG std::cout << "\tCUDAKernelCallExpr kce\n";
        	newExpr = RewriteCUDAKernelCall(kce);
        	return true;
        } else if  (CallExpr *ce = dyn_cast<CallExpr>(e)) {
        	if (ce->getDirectCallee()->getNameAsString().find("cuda") == 0) {
        		//DEBUG std::cout << "\tCallExpr ce "<< ce->getDirectCallee()->getNameAsString() << "\n";
        		return RewriteCUDACall(ce, newExpr);
        	}
        	else { //Common function call, fixme default value parameters
        	}
        } else if (MemberExpr *me = dyn_cast<MemberExpr>(e)) {
        	//DEBUG std::cout << "\tMemberExpr me\n";
        	//Catches expressions which refer to the member of a struct or class
        	// in the CUDA case these are primarily just dim3s and cudaDeviceProp
        } else if (ExplicitCastExpr *ece = dyn_cast<ExplicitCastExpr>(e)) {
        	//DEBUG std::cout << "\tExplicitCastExpr ece\n";
        	//Rewrite explicit casts of CUDA data types
        } else if (UnaryExprOrTypeTraitExpr *soe = dyn_cast<UnaryExprOrTypeTraitExpr>(e)) {
        	//DEBUG std::cout << "\tUnaryExprOrTypeTraitExpr soe\n";
        	//Rewrite unary expressions or type trait expressions (things like sizeof)
        } else if (CXXTemporaryObjectExpr *cte = dyn_cast<CXXTemporaryObjectExpr>(e)) {
        	//DEBUG std::cout << "\tCXXTemporaryObjectExpr cte\n";
        	//Catches dim3 declarations of the form: some_var=dim3(x,y,z);
        	// the RHS is considered a temporary object
        } else if (CXXConstructExpr *cce = dyn_cast<CXXConstructExpr>(e)) {
        	//DEBUG std::cout << "\tCXXConstructExpr cce\n";
        	//Catches dim3 declarations of the form: dim3 some_var(x,y,z);
        }
        bool ret = false;
        //Do a DFS, recursing into children, then rewriting this expression
        //if rewrite happened, replace text at old sourcerange
        for (Stmt::child_iterator CI = e->child_begin(), CE = e->child_end(); CI != CE; ++CI) {
            std::string s;
            Expr *child = (Expr *) *CI;
            if (child && RewriteHostExpr(child, s)) {
            	//DEBUG std::cout << "I was a child?\n";
                //Perform "rewrite", which is just a simple replace
                ReplaceStmtWithText(child, s, exprRewriter);
                ret = true;
            }
        }
        newExpr = exprRewriter.getRewrittenText(realRange);
        //Rew->ReplaceText(realRange, newExpr);

//        SourceRange newrealRange(SM->getExpansionLoc(e->getLocStart()),
//                              SM->getExpansionLoc(e->getLocEnd()));
//        newExpr = exprRewriter.getRewrittenText(realRange);
        return ret;
    }

    std::string RewriteCUDAKernelCall(clang::CUDAKernelCallExpr *kernelCall) {
    	//DEBUG SourceLocation sloc = SM->getSpellingLoc(kernelCall->getExprLoc());
    	//DEBUG std::cout << "RewriteCUDAKernelCall kernelCall: " << sloc.printToString(*SM) << "\n";
        CallExpr *kernelConfig = kernelCall->getConfig();
        Expr *grid = kernelConfig->getArg(0);
        Expr *block = kernelConfig->getArg(1);
        FunctionDecl* callee = kernelCall->getDirectCallee();
//        std::cout << PrintDeclToString(callee) << "\n"; //
//        std::cout << getStmtText(kernelCall) << "\n";
//        std::cout << getStmtText(kernelCall->getCallee()) << "\n";
//        std::cout << getStmtText(block) << "\n"; //g
//        std::cout << getStmtText(kernelCall->getArg(0)) << "\n"; //42

        //TODO Check if this is a general rule
        std::string SStr;
        llvm::raw_string_ostream S(SStr);
        S << getStmtText(kernelCall->getCallee()) << "(";
        for(int i = 0; i < kernelCall->getNumArgs(); i++){
        	S << getStmtText(kernelCall->getArg(i)) << ", ";
        }
        //std::cout << getStmtText(kernelCall->getArg(kernelCall->getNumArgs()-1)) << "
        S << getStmtText(grid) << ", " << getStmtText(block) << ", " << "TODO: INSERT BLOCKIDX VAR" << ");";
        return S.str();
    }

    bool RewriteCUDACall(CallExpr *cudaCall, std::string &newExpr) {
    	//DEBUG SourceLocation sloc = SM->getSpellingLoc(cudaCall->getExprLoc());
    	//DEBUG std::cout << "RewriteCUDACall cudaCall: " << sloc.printToString(*SM) << "\n";
        std::string funcName = cudaCall->getDirectCallee()->getNameAsString();
        //TODO we have to match funcName.
        std::string a = " //cuda api translation required";
        newExpr = funcName + a;
        return true;
    }



    //The workhorse that takes the constructed replacement attribute and inserts it in place of the old one
    void RewriteAttr(Attr *attr, std::string replace, Rewriter &rewrite){
    	//DEBUG std::cout << "RewriteAttr!\n";
        SourceLocation instLoc = SM->getExpansionLoc(attr->getLocation());
        SourceRange realRange(instLoc, PP->getLocForEndOfToken(instLoc));
        rewrite.ReplaceText(instLoc, rewrite.getRangeSize(realRange), replace);
    }


    std::string getStmtText(Stmt *s) {
        SourceLocation a(SM->getExpansionLoc(s->getLocStart())), b(Lexer::getLocForEndOfToken(SourceLocation(SM->getExpansionLoc(s->getLocEnd())), 0,  *SM, *LO));
        return std::string(SM->getCharacterData(a), SM->getCharacterData(b)-SM->getCharacterData(a));
    }

    //DEPRECATED: Old method to get the string representation of a Stmt
       std::string PrintStmtToString(Stmt *s) {
           std::string SStr;
           llvm::raw_string_ostream S(SStr);
           s->printPretty(S, 0, PrintingPolicy(*LO));
           return S.str();
       }

       //DEPRECATED: Old method to get the string representation of a Decl
       //TODO: Test replacing the one remaining usage in HandleTranslationUnit with getStmtText
       std::string PrintDeclToString(Decl *d) {
           std::string SStr;
           llvm::raw_string_ostream S(SStr);
           d->print(S);
           return S.str();
       }

       //Replace a chunk of code represented by a Stmt with a constructed string
       bool ReplaceStmtWithText(Stmt *OldStmt, llvm::StringRef NewStr, Rewriter &Rewrite) {
           SourceRange origRange = OldStmt->getSourceRange();
           SourceLocation s = SM->getExpansionLoc(origRange.getBegin());
           SourceLocation e = PP->getLocForEndOfToken(SM->getExpansionLoc(origRange.getEnd()));
           return Rewrite.ReplaceText(s,
                                      Rewrite.getRangeSize(SourceRange(s, e)),
                                      NewStr);
       }
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(&CI, &TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

//class Replication : public ASTFrontendAction{
//public:
//	Replication(){}
//	void EndSourceFileAction() override {
//		TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs()); //write in a temporary (file, stream?)
//
//	}
//
//	std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
//		TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
//		return llvm::make_unique<MyReplicator>(&CI, &TheRewriter);
//	}
//private:
//	Rewriter TheRewriter;
//};



int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());
  //Tool.run(newFrontendActionFactory<Replication>().get());
  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
