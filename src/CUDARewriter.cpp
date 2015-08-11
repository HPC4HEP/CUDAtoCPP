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

        //Rewrite formal parameters
        //for (FunctionDecl::param_iterator PI = kf->param_begin(), PE = kf->param_end(); PI != PE; ++PI) {
        	//DEBUG std::cout << "\tParam! \n";
        	//std::cout << kf->getParamDecl(0)->getType().getAsString() << kf->getParamDecl(0)->getQualifiedNameAsString() << "\n";
        	//kf->getParamDecl(0)->getQualifiedNameAsString()
//            RewriteKernelParam(*PI, kf->hasAttr<CUDAGlobalAttr>());
        //}

        //Parameter Rewriting for a kernel
    	if (kf->hasAttr<CUDAGlobalAttr>()) {
    		//fixme doens't enters here, why?
    		//DEBUG std::cout << "\tkf->hasAttr<CUDAGlobalAttr>() = true\n";
        	//Means host callable
    		std::string SStr;
			llvm::raw_string_ostream S(SStr);
			S << kf->getCallResultType().getAsString() << " " << kf->getNameAsString() << "(";
			for( int j = 0; j < kf->getNumParams(); j++){
				//TODO Check if this is a general rule
				S << kf->getParamDecl(j)->getType().getAsString() << " " << kf->getParamDecl(0)->getQualifiedNameAsString() << ", ";
			}
			S << "dim3 blockDim, dim3 gridDim)";
			//DEBUG: std::cout << S.str() << "\n";
			SourceLocation start = SM->getExpansionLoc(kf->getLocStart());
			SourceLocation end = PP->getLocForEndOfToken(SM->getExpansionLoc(kf->getParamDecl(kf->getNumParams()-1)->getLocEnd()));
			SourceRange range(start, end);//PP->getLocForEndOfToken(instLoc));
			Rew->ReplaceText(start, Rew->getRangeSize(range), S.str());


        }
        if (kf->hasBody()) {
        	//DEBUG std::cout << "\tkf->hasBody() = true\n";
            RewriteKernelStmt(kf->getBody());
        }


    }
//    void RewriteKernelParam(ParmVarDecl *parmDecl, bool isFuncGlobal) {
//    	//DEBUG SourceLocation sloc = SM->getSpellingLoc(parmDecl->getLocation());
//    	//DEBUG std::cout << "RewriteKernelParam parmDecl: " << sloc.printToString(*SM) << "\n";
//    	//TODO Formal parameters transformation should happen here.
//    	//	TODO add blockDim, gridDim, blockIdx?
//
//
//    }

    void RewriteKernelStmt(Stmt *ks) {
    	//DEBUG std::cout << "RewriteKernelStmt ks: ";
        //Visit this node
		if (Expr *e = dyn_cast<Expr>(ks)) {
			//DEBUG SourceLocation sloc = SM->getSpellingLoc(e->getExprLoc());
			//DEBUG std::cout << "<Expr> " << sloc.printToString(*SM) << "\n";
			std::string str;
			//FIXME it has to be shaped like this? Why RewriteKernelExpr() cannot just directly rewrite?
			if (RewriteKernelExpr(e, str)) {
				//ReplaceStmtWithText(e, str, KernelRewrite);
			}
		}
		else if (DeclStmt *ds = dyn_cast<DeclStmt>(ks)) {
			//DEBUG SourceLocation sloc = SM->getSpellingLoc(ds->getStartLoc());
			//DEBUG std::cout << "<DeclStmt> " << sloc.printToString(*SM) << "\n";
			DeclGroupRef DG = ds->getDeclGroup();
			for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
				if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
					//DEBUG std::cout << "\tVarDecl vd\n";
					RewriteKernelVarDecl(vd);
				}
				//TODO other non-top level declarations??
			}
		}
		//TODO rewrite any other Stmts?

		else {
			//Traverse children and recurse
			for (Stmt::child_iterator CI = ks->child_begin(), CE = ks->child_end(); CI != CE; ++CI) {
				//DEBUG std::cout << "children!\n";
				if (*CI) RewriteKernelStmt(*CI);
			}
		}
    }

    bool RewriteKernelExpr(Expr *e, std::string &newExpr) {
    	//DEBUG SourceLocation sloc = SM->getSpellingLoc(e->getExprLoc());
    	//DEBUG std::cout << "RewriteKernelExpr e: " << sloc.printToString(*SM) << "\n";
    	//TODO Loop Fission algorithm (Stage 2) should be triggered here
    	//TODO but how to see everything in the scope of the __syncthreads() expr?
    	return true;
    }

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
    	SourceLocation sloc = SM->getSpellingLoc(e->getExprLoc());
    	std::cout << "RewriteHostExpr e: " << sloc.printToString(*SM) << "\n";

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
        		std::cout << "\tCall?!?!?\n";
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
        S << getStmtText(grid) << ", " << getStmtText(block) << ")";
        return S.str();
    }

    bool RewriteCUDACall(CallExpr *cudaCall, std::string &newExpr) {
    	//DEBUG SourceLocation sloc = SM->getSpellingLoc(cudaCall->getExprLoc());
    	//DEBUG std::cout << "RewriteCUDACall cudaCall: " << sloc.printToString(*SM) << "\n";
        std::string funcName = cudaCall->getDirectCallee()->getNameAsString();
        //TODO we have to match funcName.
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
           SourceLocation e = SM->getExpansionLoc(origRange.getEnd());
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

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
